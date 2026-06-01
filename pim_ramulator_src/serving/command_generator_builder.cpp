#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "base/exception.h"
#include "frontend/impl/serving/command_generator_internal.h"

namespace Ramulator::Serving {

namespace {

// LPDDR5-bank program builder
class LpddrProgramBuilder {
 public:
  explicit LpddrProgramBuilder(const GeneratorGeometry& geometry, int context_tokens)
      : m_geometry(geometry),
        m_context_tokens(std::max(1, context_tokens)),
        m_dhead(std::max(1, geometry.dhead)),
        m_heads_per_hbm(std::max(1, geometry.heads_per_hbm)),
        m_valid_channel_limit(std::max(1, geometry.channel_count)),
        m_dtype_bytes(std::max(1, geometry.dtype_bytes)),
        m_n_mac(static_cast<int>(kPrefetchBytes / m_dtype_bytes)) {}

  GeneratedHybridArtifacts build(uint64_t page_size_bytes) {
    if (page_size_bytes != 4096) {
      throw ConfigurationError("build_lpddr5_bank_generated_artifacts requires 4KB pages");
    }
    build_iterations();
    build_barrier();
    append_program();
    return finalize(page_size_bytes);
  }

 private:
  int score_mac_depth() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_context_tokens) /
                                      static_cast<double>(kNPseudoChannels * kNRank * kNBankGroup)));
  }

  int score_mac_width() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_dhead) /
                                      static_cast<double>(kNBank * m_n_mac)));
  }

  int score_windows() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_context_tokens) /
                                      static_cast<double>(kNPseudoChannels * kNRank *
                                                          kNBankGroup * 16)));
  }

  int context_mvgb_cols() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_context_tokens) /
                                      static_cast<double>(kNPseudoChannels * kNRank *
                                                          kNBankGroup * m_n_mac)));
  }

  int context_mac_depth() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_dhead) /
                                      static_cast<double>(kNBank * m_n_mac)));
  }

  int context_mac_width() const {
    return static_cast<int>(std::ceil(static_cast<double>(m_context_tokens) /
                                      static_cast<double>(kNPseudoChannels * kNRank * kNBankGroup)));
  }

  int iteration_channel_count(int itr) const {
    const int consumed = (itr + 1) * m_valid_channel_limit;
    if (consumed < m_heads_per_hbm) {
      return m_valid_channel_limit;
    }
    const int remainder = m_heads_per_hbm % m_valid_channel_limit;
    return remainder == 0 ? m_valid_channel_limit : remainder;
  }

  void append_commands(const CommandList& commands) {
    m_total_cmd.insert(m_total_cmd.end(), commands.begin(), commands.end());
  }

  void append_group(const CommandGroups& groups, int group_idx) {
    if (group_idx < 0 || group_idx >= static_cast<int>(groups.size())) {
      return;
    }
    append_commands(groups[group_idx]);
  }

  void append_barrier() {
    append_commands(m_barrier);
  }

  void build_iterations() {
    m_num_itr = static_cast<int>(std::ceil(static_cast<double>(m_heads_per_hbm) /
                                           static_cast<double>(m_valid_channel_limit)));
    for (int itr = 0; itr < m_num_itr; ++itr) {
      build_iteration(itr, iteration_channel_count(itr));
    }
  }

  void build_iteration(int itr, int per_itr_channels) {
    m_cmd_score_wrgb.emplace_back();
    m_cmd_score_mac.emplace_back();
    m_cmd_score_mvsb.emplace_back();
    m_cmd_sfm.emplace_back();
    m_cmd_context_mvgb.emplace_back();
    m_cmd_context_mac.emplace_back();
    m_cmd_context_mvsb.emplace_back();
    m_valid_channels.push_back(per_itr_channels);

    build_score_phase(itr, per_itr_channels);
    build_softmax_phase(itr, per_itr_channels);
    build_context_phase(itr, per_itr_channels);
  }

  // Score phase
  void build_score_phase(int itr, int per_itr_channels) {
    for (int ba_idx = 0; ba_idx < kNBank; ++ba_idx) {
      for (int col_idx = 0; col_idx < score_mac_width(); ++col_idx) {
        for (int lch = 0; lch < per_itr_channels; ++lch) {
          const uint64_t slice_relative =
              static_cast<uint64_t>(ba_idx) * m_gs.ba + static_cast<uint64_t>(col_idx);
          m_cmd_score_wrgb[itr].push_back(make_kv_command(
              Request::Type::PIM_WR_GB,
              CommandTargetKind::KvKey,
              make_kv_address(itr, lch, slice_relative, m_gs)));
        }
      }
    }

    for (int n_idx = 0; n_idx < score_mac_depth(); ++n_idx) {
      m_cmd_score_mac[itr].emplace_back();
      for (int k_idx = 0; k_idx < score_mac_width(); ++k_idx) {
        const int idx = k_idx + n_idx * score_mac_width();
        for (int lch = 0; lch < per_itr_channels; ++lch) {
          const uint64_t slice_relative = static_cast<uint64_t>(idx) * m_gs.col;
          m_cmd_score_mac[itr].back().push_back(make_kv_command(
              Request::Type::PIM_MAC_AB,
              CommandTargetKind::KvKey,
              make_kv_address(itr, lch, slice_relative, m_gs)));
        }
      }

      if (n_idx % 16 != 15 && n_idx != score_mac_depth() - 1) {
        continue;
      }

      m_cmd_score_mvsb[itr].emplace_back();
      for (int bg_idx = 0; bg_idx < kNBankGroup; ++bg_idx) {
        for (int rank = 0; rank < kNRank; ++rank) {
          for (int lch = 0; lch < per_itr_channels; ++lch) {
            const uint64_t slice_relative = static_cast<uint64_t>(rank) * m_gs.rank +
                                            static_cast<uint64_t>(bg_idx) * m_gs.bg;
            m_cmd_score_mvsb[itr].back().push_back(make_kv_command(
                Request::Type::PIM_MV_SB,
                CommandTargetKind::KvKey,
                make_kv_address(itr, lch, slice_relative, m_gs)));
          }
        }
      }
    }
  }

  // Softmax phase
  void build_softmax_phase(int itr, int per_itr_channels) {
    for (int lch = 0; lch < per_itr_channels; ++lch) {
      m_cmd_sfm[itr].push_back(make_passthrough_command(
          Request::Type::PIM_SFM, static_cast<uint64_t>(lch) * m_gs.ch));
    }
  }

  // Context phase
  void build_context_phase(int itr, int per_itr_channels) {
    for (int rank = 0; rank < kNRank; ++rank) {
      for (int bg_idx = 0; bg_idx < kNBankGroup; ++bg_idx) {
        for (int col_idx = 0; col_idx < context_mvgb_cols(); ++col_idx) {
          for (int lch = 0; lch < per_itr_channels; ++lch) {
            const uint64_t slice_relative = static_cast<uint64_t>(rank) * m_gs.rank +
                                            static_cast<uint64_t>(bg_idx) * m_gs.bg +
                                            static_cast<uint64_t>(col_idx);
            m_cmd_context_mvgb[itr].push_back(make_kv_command(
                Request::Type::PIM_MV_GB,
                CommandTargetKind::KvValue,
                make_kv_address(itr, lch, slice_relative, m_gs)));
          }
        }
      }
    }

    for (int n_idx = 0; n_idx < context_mac_depth(); ++n_idx) {
      m_cmd_context_mac[itr].emplace_back();
      for (int k_idx = 0; k_idx < context_mac_width(); ++k_idx) {
        const int idx = k_idx + n_idx * context_mac_width();
        for (int lch = 0; lch < per_itr_channels; ++lch) {
          const uint64_t slice_relative = static_cast<uint64_t>(idx) * m_gs.col;
          m_cmd_context_mac[itr].back().push_back(make_kv_command(
              Request::Type::PIM_MAC_AB,
              CommandTargetKind::KvValue,
              make_kv_address(itr, lch, slice_relative, m_gs)));
        }
      }

      m_cmd_context_mvsb[itr].emplace_back();
      for (int ba_idx = 0; ba_idx < kNBank; ++ba_idx) {
        for (int rank = 0; rank < kNRank; ++rank) {
          for (int lch = 0; lch < per_itr_channels; ++lch) {
            const uint64_t slice_relative = static_cast<uint64_t>(rank) * m_gs.rank +
                                            static_cast<uint64_t>(ba_idx) * m_gs.ba;
            m_cmd_context_mvsb[itr].back().push_back(make_kv_command(
                Request::Type::PIM_MV_SB,
                CommandTargetKind::KvValue,
                make_kv_address(itr, lch, slice_relative, m_gs)));
          }
        }
      }
    }
  }

  void build_barrier() {
    for (int lch = 0; lch < m_valid_channel_limit; ++lch) {
      m_barrier.push_back(make_passthrough_command(Request::Type::PIM_BARRIER,
                                                   static_cast<uint64_t>(lch) * m_gs.ch));
    }
  }

  int score_wrgb_stride_for(int next_itr) const {
    return static_cast<int>(
        kNBank * std::ceil(static_cast<double>(m_dhead) / static_cast<double>(kNBank * m_n_mac)) *
        std::ceil(static_cast<double>(m_valid_channels[next_itr])) /
        static_cast<double>(score_windows()));
  }

  int context_mvgb_stride_for(int itr, int denom) const {
    return static_cast<int>(
        kNRank * kNBankGroup *
        std::ceil(static_cast<double>(m_context_tokens) /
                  static_cast<double>(kNPseudoChannels * kNRank * kNBankGroup * m_n_mac)) *
        std::ceil(static_cast<double>(m_valid_channels[itr])) / static_cast<double>(denom));
  }

  // Pair iterations
  void append_paired_iteration(int i) {
    append_commands(m_cmd_score_wrgb[i]);
    if (i == 0 && !m_cmd_score_mac[i].empty()) {
      for (int j = 0; j < m_valid_channels[i] && j < static_cast<int>(m_cmd_score_mac[i][0].size());
           ++j) {
        m_total_cmd.push_back(m_cmd_score_mac[i][0][j]);
      }
    }
    append_barrier();

    for (int j = 0; j < score_windows() + 1; ++j) {
      if (j != score_windows()) {
        for (int k = 0; k < 16; ++k) {
          append_group(m_cmd_score_mac[i], j * 16 + k);
        }
      }
      if (j != 0) {
        append_group(m_cmd_score_mvsb[i], j - 1);
      }
      if (j != score_windows()) {
        const int stride = score_wrgb_stride_for(i + 1);
        for (int k = 0; k < stride; ++k) {
          const int pos = j * stride + k;
          if (pos >= static_cast<int>(m_cmd_score_wrgb[i + 1].size())) {
            break;
          }
          m_total_cmd.push_back(m_cmd_score_wrgb[i + 1][pos]);
        }
        append_barrier();
      }
    }

    const int score_mvgb_start = static_cast<int>(std::floor(static_cast<double>(score_windows()) / 2.0));
    const int score_mvgb_denom =
        std::max(1, static_cast<int>(std::ceil(static_cast<double>(score_windows()) / 2.0)));
    for (int j = 0; j < score_windows() + 1; ++j) {
      if (j != score_windows()) {
        for (int k = 0; k < 16; ++k) {
          append_group(m_cmd_score_mac[i + 1], j * 16 + k);
        }
      }
      if (j != 0) {
        append_group(m_cmd_score_mvsb[i + 1], j - 1);
      }
      if (j == 0) {
        append_commands(m_cmd_sfm[i]);
      }
      if (j != score_windows() && j >= score_mvgb_start) {
        const int stride = context_mvgb_stride_for(i, score_mvgb_denom);
        for (int k = 0; k < stride; ++k) {
          const int pos = (j - score_mvgb_start) * stride + k;
          if (pos >= static_cast<int>(m_cmd_context_mvgb[i].size())) {
            break;
          }
          m_total_cmd.push_back(m_cmd_context_mvgb[i][pos]);
        }
      }
      if (j != score_windows()) {
        append_barrier();
      }
    }

    const int context_mvgb_start =
        static_cast<int>(std::floor(static_cast<double>(context_mac_depth()) / 2.0));
    const int context_mvgb_denom =
        std::max(1, static_cast<int>(std::ceil(static_cast<double>(context_mac_depth()) / 2.0)));

    for (int j = 0; j < context_mac_depth() + 1; ++j) {
      if (j != context_mac_depth()) {
        append_group(m_cmd_context_mac[i], j);
      }
      if (j != 0) {
        append_group(m_cmd_context_mvsb[i], j - 1);
      }
      if (j == 0) {
        append_commands(m_cmd_sfm[i + 1]);
      }
      if (j != context_mac_depth() && j >= context_mvgb_start) {
        const int stride = context_mvgb_stride_for(i + 1, context_mvgb_denom);
        for (int k = 0; k < stride; ++k) {
          const int pos = (j - context_mvgb_start) * stride + k;
          if (pos >= static_cast<int>(m_cmd_context_mvgb[i + 1].size())) {
            break;
          }
          m_total_cmd.push_back(m_cmd_context_mvgb[i + 1][pos]);
        }
      }
      if (j != context_mac_depth()) {
        append_barrier();
      }
    }

    for (int j = 0; j < context_mac_depth() + 1; ++j) {
      if (j != context_mac_depth()) {
        append_group(m_cmd_context_mac[i], j);
      }
      if (j != 0) {
        append_group(m_cmd_context_mvsb[i], j - 1);
      }
      if (j != context_mac_depth()) {
        append_barrier();
      }
    }
  }

  // Tail iteration
  void append_tail_iteration() {
    if (m_num_itr % 2 == 0) {
      return;
    }

    const int i = m_num_itr - 1;
    append_commands(m_cmd_score_wrgb[i]);
    append_barrier();

    for (int j = 0; j < score_windows() + 1; ++j) {
      if (j != score_windows()) {
        for (int k = 0; k < 16; ++k) {
          append_group(m_cmd_score_mac[i], j * 16 + k);
        }
      }
      if (j != 0) {
        append_group(m_cmd_score_mvsb[i], j - 1);
      }
      if (j != score_windows()) {
        append_barrier();
      }
    }

    append_commands(m_cmd_sfm[i]);
    append_commands(m_cmd_context_mvgb[i]);
    append_barrier();

    for (int j = 0; j < context_mac_depth() + 1; ++j) {
      if (j != context_mac_depth()) {
        append_group(m_cmd_context_mac[i], j);
      }
      if (j != 0) {
        append_group(m_cmd_context_mvsb[i], j - 1);
      }
      if (j != context_mac_depth()) {
        append_barrier();
      }
    }
  }

  void append_program() {
    for (int i = 0; i < m_num_itr - 1; i += 2) {
      append_paired_iteration(i);
    }
    append_tail_iteration();
  }

  GeneratedHybridArtifacts finalize(uint64_t page_size_bytes) {
    auto layout = std::make_shared<RepresentativeKvLayout>();
    layout->geometry = m_geometry;
    layout->context_tokens = m_context_tokens;
    layout->page_size_bytes = page_size_bytes;
    collect_compact_layout_from_commands(m_total_cmd, *layout);

    GeneratedHybridArtifacts artifacts;
    artifacts.commands = std::move(m_total_cmd);
    artifacts.kv_layout = std::move(layout);
    return artifacts;
  }

  GeneratorGeometry m_geometry;
  int m_context_tokens = 1;
  int m_dhead = 1;
  int m_heads_per_hbm = 1;
  int m_valid_channel_limit = 1;
  int m_dtype_bytes = 1;
  int m_n_mac = 1;
  int m_num_itr = 0;
  LpddrGeometry m_gs;

  std::vector<CommandList> m_cmd_score_wrgb;
  std::vector<CommandGroups> m_cmd_score_mac;
  std::vector<CommandGroups> m_cmd_score_mvsb;
  std::vector<CommandList> m_cmd_sfm;
  std::vector<CommandList> m_cmd_context_mvgb;
  std::vector<CommandGroups> m_cmd_context_mac;
  std::vector<CommandGroups> m_cmd_context_mvsb;
  std::vector<int> m_valid_channels;
  CommandList m_barrier;
  CommandList m_total_cmd;
};

}  // namespace

GeneratedHybridArtifacts build_lpddr5_bank_generated_artifacts(const GeneratorGeometry& geometry,
                                                               int context_tokens,
                                                               uint64_t page_size_bytes) {
  LpddrProgramBuilder builder(geometry, context_tokens);
  return builder.build(page_size_bytes);
}

std::vector<LogicalCommand> build_lpddr5_bank_logical_trace(const GeneratorGeometry& geometry,
                                                            int context_tokens) {
  return build_lpddr5_bank_generated_artifacts(geometry, context_tokens, 4096).commands;
}

}  // namespace Ramulator::Serving
