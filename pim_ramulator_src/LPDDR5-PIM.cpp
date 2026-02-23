#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class LPDDR5PIM : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, LPDDR5PIM, "LPDDR5-PIM", "LPDDR5-PIM Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      //   name           density   DQ   Ch Ra Bg Ba   Ro     Co
      {"LPDDR5_2Gb_x16",  {2<<10,   16, {1, 1, 4, 4, 1<<13, 1<<10}}},
      {"LPDDR5_4Gb_x16",  {4<<10,   16, {1, 1, 4, 4, 1<<14, 1<<10}}},
      {"LPDDR5_8Gb_x16",  {8<<10,   16, {1, 1, 4, 4, 1<<15, 1<<10}}},
      {"LPDDR5_16Gb_x16", {16<<10,  16, {1, 1, 4, 4, 1<<16, 1<<10}}},
      {"LPDDR5_32Gb_x16", {32<<10,  16, {1, 1, 4, 4, 1<<17, 1<<10}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name         rate   nBL  nCL  nRCD  nRPab  nRPpb   nRAS  nRC   nWR  nRTP nCWL nCCD nCCDAB nCCDSB nRRD nWTRS nWTRL nFAW  nPPD  nRFCab nRFCpb nREFI nPBR2PBR nPBR2ACT nCS,  tCK_ps
      {"LPDDR5_6400",  {6400,  4,   20,   15,    17,   15,     34,   30,   28,   4,  11,   4,    4,     4,    4,   5,    10,   16,  2,   -1,      -1,   -1,   -1,        -1,    2,   1250}},
    };


  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 8;

    inline static constexpr ImplDef m_levels = {
      "channel", "rank", "bankgroup", "bank", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      // LPDDR5 commands
      "ACT-1",  "ACT-2",
      "PRE",    "PREA",
      "CASRD",  "CASWR",   // WCK2CK Sync
      "RD16",   "WR16",   "RD16A",   "WR16A",
      "REFab",  "REFpb",
      "RFMab",  "RFMpb",
      // PIM commands
      "ACTAB", "ACTSB", "ACTPB",
      "MACAB", "MACSB", "MACPB",
      "WRGB", "MVSB", "MVGB", "SFM",
      "SETM", "SETH", "BARRIER",
      // PIM helper precharge commands (non-JEDEC helpers)
      "PRESB", "PREPB",
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT-1", "row"},    {"ACT-2",  "row"},
        {"PRE",   "bank"},   {"PREA",   "rank"},
        {"CASRD", "rank"},   {"CASWR",  "rank"},
        {"RD16",  "column"}, {"WR16",   "column"}, {"RD16A", "column"}, {"WR16A", "column"},
        {"REFab", "rank"},   {"REFpb",  "rank"},
        {"RFMab", "rank"},   {"RFMpb",  "rank"},
        // PIM command scopes
        {"ACTAB", "row"}, {"ACTSB", "row"}, {"ACTPB", "row"},
        {"MACAB", "column"}, {"MACSB", "column"}, {"MACPB", "column"},
        {"WRGB",  "bank"},   {"MVSB",  "bank"},   {"MVGB",  "bank"}, {"SFM", "rank"},
        {"SETM",  "bank"},   {"SETH",  "channel"}, {"BARRIER", "channel"},
        {"PRESB", "bank"},   {"PREPB", "bank"},
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT-1",  {false,  false,   false,   false}},
        {"ACT-2",  {true,   false,   false,   false}},
        {"PRE",    {false,  true,    false,   false}},
        {"PREA",   {false,  true,    false,   false}},
        {"CASRD",  {false,  false,   false,   false}},
        {"CASWR",  {false,  false,   false,   false}},
        {"RD16",   {false,  false,   true,    false}},
        {"WR16",   {false,  false,   true,    false}},
        {"RD16A",  {false,  true,    true,    false}},
        {"WR16A",  {false,  true,    true,    false}},
        {"REFab",  {false,  false,   false,   true }},
        {"REFpb",  {false,  false,   false,   true }},
        {"RFMab",  {false,  false,   false,   true }},
        {"RFMpb",  {false,  false,   false,   true }},
        {"ACTAB",  {true,   false,   false,   false}},
        {"ACTSB",  {true,   false,   false,   false}},
        {"ACTPB",  {true,   false,   false,   false}},
        {"MACAB",  {false,  false,   true,    false}},
        {"MACSB",  {false,  false,   true,    false}},
        {"MACPB",  {false,  false,   true,    false}},
        {"WRGB",   {false,  false,   false,   false}},
        {"MVSB",   {false,  false,   false,   false}},
        {"MVGB",   {false,  false,   false,   false}},
        {"SFM",    {false,  false,   false,   false}},
        {"SETM",   {false,  false,   false,   false}},
        {"SETH",   {false,  false,   false,   false}},
        {"BARRIER",{false,  false,   false,   false}},
        {"PRESB",  {false,  true,    false,   false}},
        {"PREPB",  {false,  true,    false,   false}},
      }
    );

    inline static constexpr ImplDef m_requests = {
      // LPDDR5 requests
      "read16", "write16",
      "all-bank-refresh", "per-bank-refresh",
      // PIM requests
      "pim-mac-all-bank", "pim-mac-same-bank", "pim-mac-per-bank",
      "pim-write-to-gemv-buffer", "pim-move-to-softmax-buffer", "pim-move-to-gemv-buffer",
      "pim-softmax", "pim-set-model", "pim-set-head", "pim-barrier",
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read16", "RD16"}, {"write16", "WR16"},
        {"all-bank-refresh", "REFab"}, {"per-bank-refresh", "REFpb"},
        {"pim-mac-all-bank", "MACAB"}, {"pim-mac-same-bank", "MACSB"}, {"pim-mac-per-bank", "MACPB"},
        {"pim-write-to-gemv-buffer", "WRGB"}, {"pim-move-to-softmax-buffer", "MVSB"}, {"pim-move-to-gemv-buffer", "MVGB"},
        {"pim-softmax", "SFM"}, {"pim-set-model", "SETM"}, {"pim-set-head", "SETH"}, {"pim-barrier", "BARRIER"},
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL16", "nCL", "nRCD", "nRPab", "nRPpb", "nRAS", "nRC", "nWR", "nRTP", "nCWL",
      "nCCD", "nCCDAB", "nCCDSB",
      "nRRD",
      "nWTRS", "nWTRL",
      "nFAW",
      "nPPD",
      "nRFCab", "nRFCpb","nREFI",
      "nPBR2PBR", "nPBR2ACT",
      "nCS",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
    //    ACT-1       ACT-2
       "Pre-Opened", "Opened", "Closed", "PowerUp", "N/A"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",   "N/A"}, 
        {"rank",      "PowerUp"},
        {"bankgroup", "N/A"},
        {"bank",      "Closed"},
        {"row",       "Closed"},
        {"column",    "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<LPDDR5PIM> {
      Clk_t m_final_synced_cycle = -1; // Extra CAS Sync command needed for RD/WR after this cycle

      Node(LPDDR5PIM* dram, Node* parent, int level, int id) : DRAMNodeBase<LPDDR5PIM>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(32);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      // Sanity check: is the calculated chip density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} chip density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 2);
      m_timing_vals("tCK_ps") = tCK_ps;

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFCab_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          130,  180,  210,  280, 
      };

      constexpr int tRFCpb_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          60,   90,   120,  140, 
      };

      constexpr int tPBR2PBR_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          60,   90,   90,  90, 
      };

      constexpr int tPBR2ACT_TABLE[4] = {
      //  2Gb   4Gb   8Gb  16Gb
          8,    8,    8,   8, 
      };

      // tREFI(base) table (unit is nanosecond!)
      constexpr int tREFI_BASE = 3906;
      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 2048:  return 0;
          case 4096:  return 1;
          case 8192:  return 2;
          case 16384: return 3;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFCab")    = JEDEC_rounding(tRFCab_TABLE[density_id], tCK_ps);
      m_timing_vals("nRFCpb")    = JEDEC_rounding(tRFCpb_TABLE[density_id], tCK_ps);
      m_timing_vals("nPBR2PBR")  = JEDEC_rounding(tPBR2PBR_TABLE[density_id], tCK_ps);
      m_timing_vals("nPBR2ACT")  = JEDEC_rounding(tPBR2ACT_TABLE[density_id], tCK_ps);
      m_timing_vals("nREFI") = JEDEC_rounding(tREFI_BASE, tCK_ps);

      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL16");

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          /////////////////////////////////
          ////--         PIM           --//
          /////////////////////////////////

          /*** PIM-MAC-All-Bank ***/
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"ACTAB", "ACT-1", "PREA", "REFab", "REFpb"}, .latency = V("nRRD")},
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"ACTAB"}, .latency = V("nFAW"), .window = 4},
          {.level = "rank", .preceding = {"MACAB"}, .following = {"MACAB"}, .latency = V("nCCDAB")},
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"MACAB"}, .latency = V("nRCD")},
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"PREA"}, .latency = V("nRAS")},
          {.level = "rank", .preceding = {"MACAB"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "rank", .preceding = {"PREA"}, .following = {"ACTAB"}, .latency = V("nRPab")},
          {.level = "rank", .preceding = {"ACTAB"}, .following = {"REFab"}, .latency = V("nRC")},
          {.level = "rank", .preceding = {"PREA"}, .following = {"REFab"}, .latency = V("nRPab")},
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTAB"}, .latency = V("nRFCab")},

          /*** PIM-MAC-Same-Bank ***/
          {.level = "rank", .preceding = {"ACTSB"}, .following = {"ACTSB", "ACT-1", "PREA", "PRESB", "REFab", "REFpb"}, .latency = V("nRRD")},
          {.level = "rank", .preceding = {"ACTSB"}, .following = {"ACTSB"}, .latency = V("nFAW"), .window = 4},
          {.level = "rank", .preceding = {"MACSB"}, .following = {"MACSB"}, .latency = V("nCCDSB")},
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"ACTSB"}, .latency = V("nRC")},
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"MACSB"}, .latency = V("nRCD")},
          {.level = "bank", .preceding = {"ACTSB"}, .following = {"PRESB"}, .latency = V("nRAS")},
          {.level = "bank", .preceding = {"MACSB"}, .following = {"PRESB"}, .latency = V("nRTP")},
          {.level = "bank", .preceding = {"PRESB"}, .following = {"ACTSB"}, .latency = V("nRPpb")},
          {.level = "rank", .preceding = {"ACTSB"}, .following = {"REFab"}, .latency = V("nRC")},
          {.level = "rank", .preceding = {"PRESB"}, .following = {"REFab"}, .latency = V("nRPpb")},
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTSB"}, .latency = V("nRFCab")},

          /*** PIM-MAC-Per-Bank ***/
          {.level = "rank", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nCCD")},
          {.level = "bankgroup", .preceding = {"MACPB"}, .following = {"MACPB"}, .latency = V("nCCD")},
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"ACTPB"}, .latency = V("nRC")},
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"MACPB"}, .latency = V("nRCD")},
          {.level = "bank", .preceding = {"ACTPB"}, .following = {"PREPB"}, .latency = V("nRAS")},
          {.level = "bank", .preceding = {"MACPB"}, .following = {"PREPB"}, .latency = V("nRTP")},
          {.level = "bank", .preceding = {"PREPB"}, .following = {"ACTPB"}, .latency = V("nRPpb")},
          {.level = "rank", .preceding = {"ACTPB"}, .following = {"REFab"}, .latency = V("nRC")},
          {.level = "rank", .preceding = {"PREPB"}, .following = {"REFab"}, .latency = V("nRPpb")},
          {.level = "rank", .preceding = {"REFab"}, .following = {"ACTPB"}, .latency = V("nRFCab")},

          /*** Data Movement ***/
          {.level = "channel", .preceding = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .following = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .latency = V("nBL16")},
          {.level = "rank", .preceding = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .following = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .latency = V("nCCD")},
          {.level = "bankgroup", .preceding = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .following = {"WRGB", "MVSB", "MVGB", "SFM", "RD16", "WR16", "RD16A", "WR16A"}, .latency = V("nCCD")},

          /////////////////////////////////
          ////--     DRAM Default      --//
          /////////////////////////////////

          /*** Channel ***/ 
          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "channel", .preceding = {"RD16", "RD16A"}, .following = {"RD16", "RD16A"}, .latency = V("nBL16")},
          {.level = "channel", .preceding = {"WR16", "WR16A"}, .following = {"WR16", "WR16A"}, .latency = V("nBL16")},

          /*** Rank (or different BankGroup) ***/ 
          // CAS <-> CAS
          {.level = "rank", .preceding = {"RD16", "RD16A"}, .following = {"RD16", "RD16A"}, .latency = V("nCCD")},
          {.level = "rank", .preceding = {"WR16", "WR16A"}, .following = {"WR16", "WR16A"}, .latency = V("nCCD")},
          /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK                          
          {.level = "rank", .preceding = {"RD16", "RD16A"}, .following = {"WR16", "WR16A"}, .latency = V("nCL") + V("nBL16") + 2 - V("nCWL")},
          /// WR <-> RD, Minimum Read after Write
          {.level = "rank", .preceding = {"WR16", "WR16A"}, .following = {"RD16", "RD16A"}, .latency = V("nCWL") + V("nBL16") + V("nWTRS")},
          /// CAS <-> CAS between sibling ranks, nCS (rank switching) is needed for new DQS
          {.level = "rank", .preceding = {"RD16", "RD16A"}, .following = {"RD16", "RD16A", "WR16", "WR16A"}, .latency = V("nBL16") + V("nCS"), .is_sibling = true},
          {.level = "rank", .preceding = {"WR16", "WR16A"}, .following = {"RD16", "RD16A"}, .latency = V("nCL")  + V("nBL16") + V("nCS") - V("nCWL"), .is_sibling = true},
          /// CAS <-> PREab
          {.level = "rank", .preceding = {"RD16"}, .following = {"PREA"}, .latency = V("nRTP")},
          {.level = "rank", .preceding = {"WR16"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL16") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "rank", .preceding = {"ACT-1"}, .following = {"ACT-1", "REFpb"}, .latency = V("nRRD")},          
          {.level = "rank", .preceding = {"ACT-1"}, .following = {"ACT-1"}, .latency = V("nFAW"), .window = 4},          
          {.level = "rank", .preceding = {"ACT-1"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "rank", .preceding = {"PREA"}, .following = {"ACT-1"}, .latency = V("nRPab")},          
          /// RAS <-> REF
          {.level = "rank", .preceding = {"ACT-1"}, .following = {"REFab"}, .latency = V("nRC")},          
          {.level = "rank", .preceding = {"PRE"}, .following = {"REFab"}, .latency = V("nRPpb")},          
          {.level = "rank", .preceding = {"PREA"}, .following = {"REFab"}, .latency = V("nRPab")},          
          {.level = "rank", .preceding = {"RD16A"}, .following = {"REFab"}, .latency = V("nRPpb") + V("nRTP")},          
          {.level = "rank", .preceding = {"WR16A"}, .following = {"REFab"}, .latency = V("nCWL") + V("nBL16") + V("nWR") + V("nRPpb")},          
          {.level = "rank", .preceding = {"REFab"}, .following = {"REFab", "ACT-1", "REFpb"}, .latency = V("nRFCab")},          
          {.level = "rank", .preceding = {"ACT-1"},   .following = {"REFpb"}, .latency = V("nPBR2ACT")},  
          {.level = "rank", .preceding = {"REFpb"}, .following = {"REFpb"}, .latency = V("nPBR2PBR")},  

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD16", "RD16A"}, .following = {"RD16", "RD16A"}, .latency = V("nCCD")},          
          {.level = "bankgroup", .preceding = {"WR16", "WR16A"}, .following = {"WR16", "WR16A"}, .latency = V("nCCD")},          
          {.level = "bankgroup", .preceding = {"WR16", "WR16A"}, .following = {"RD16", "RD16A"}, .latency = V("nCWL") + V("nBL16") + V("nWTRL")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT-1"}, .following = {"ACT-1"}, .latency = V("nRRD")},  

          /*** Bank ***/ 
          {.level = "bank", .preceding = {"ACT-1"}, .following = {"ACT-1"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACT-1"}, .following = {"RD16", "RD16A", "WR16", "WR16A"}, .latency = V("nRCD")},  
          {.level = "bank", .preceding = {"ACT-1"}, .following = {"PRE"}, .latency = V("nRAS")},  
          {.level = "bank", .preceding = {"PRE"}, .following = {"ACT-1"}, .latency = V("nRPpb")},  
          {.level = "bank", .preceding = {"RD16"},  .following = {"PRE"}, .latency = V("nRTP")},  
          {.level = "bank", .preceding = {"WR16"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL16") + V("nWR")},  
          {.level = "bank", .preceding = {"RD16A"}, .following = {"ACT-1"}, .latency = V("nRTP") + V("nRPpb")},  
          {.level = "bank", .preceding = {"WR16A"}, .following = {"ACT-1"}, .latency = V("nCWL") + V("nBL16") + V("nWR") + V("nRPpb")},  
        }
      );
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Rank Actions
      m_actions[m_levels["rank"]][m_commands["PREA"]] = Lambdas::Action::Rank::PREA<LPDDR5PIM>;
      m_actions[m_levels["rank"]][m_commands["CASRD"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        node->m_final_synced_cycle = clk + m_timings["nCL"] + m_timings["nBL16"] + 1; 
      };
      m_actions[m_levels["rank"]][m_commands["CASWR"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        node->m_final_synced_cycle = clk + m_timings["nCWL"] + m_timings["nBL16"] + 1; 
      };
      m_actions[m_levels["rank"]][m_commands["RD16"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        node->m_final_synced_cycle = clk + m_timings["nCL"] + m_timings["nBL16"]; 
      };
      m_actions[m_levels["rank"]][m_commands["WR16"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        node->m_final_synced_cycle = clk + m_timings["nCWL"] + m_timings["nBL16"]; 
      };
      // Bank actions
      m_actions[m_levels["bank"]][m_commands["ACT-1"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        node->m_state = m_states["Pre-Opened"];
        node->m_row_state[target_id] = m_states["Pre-Opened"];
      };
      m_actions[m_levels["bank"]][m_commands["ACT-2"]] = Lambdas::Action::Bank::ACT<LPDDR5PIM>;
      m_actions[m_levels["bank"]][m_commands["PRE"]]   = Lambdas::Action::Bank::PRE<LPDDR5PIM>;
      m_actions[m_levels["bank"]][m_commands["RD16A"]] = Lambdas::Action::Bank::PRE<LPDDR5PIM>;
      m_actions[m_levels["bank"]][m_commands["WR16A"]] = Lambdas::Action::Bank::PRE<LPDDR5PIM>;

      // PIM helper precharges
      m_actions[m_levels["bank"]][m_commands["PREPB"]] = Lambdas::Action::Bank::PRE<LPDDR5PIM>;
      m_actions[m_levels["bank"]][m_commands["PRESB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        Node* rank = node->m_parent_node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            if (bank->m_node_id == node->m_node_id) {
              bank->m_state = m_states["Closed"];
              bank->m_row_state.clear();
            }
          }
        }
      };

      // PIM row activation (single-stage PIM commands)
      m_actions[m_levels["bank"]][m_commands["ACTPB"]] = Lambdas::Action::Bank::ACT<LPDDR5PIM>;

      m_actions[m_levels["bank"]][m_commands["ACTAB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        Node* rank = node->m_parent_node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            bank->m_state = m_states["Opened"];
            bank->m_row_state.clear();
            bank->m_row_state[target_id] = m_states["Opened"];
          }
        }
      };

      m_actions[m_levels["bank"]][m_commands["ACTSB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        Node* rank = node->m_parent_node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            if (bank->m_node_id == node->m_node_id) {
              bank->m_state = m_states["Opened"];
              bank->m_row_state.clear();
              bank->m_row_state[target_id] = m_states["Opened"];
            }
          }
        }
      };

      // Broadcast timing update across same-bank instances for MACSB.
      m_actions[m_levels["bankgroup"]][m_commands["MACSB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        AddrVec_t same_bank_addr(m_levels.size(), -1);
        same_bank_addr[m_levels["bank"]] = target_id;
        Node* rank = node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            if (bank->m_node_id == target_id) {
              bank->update_timing(cmd, same_bank_addr, clk);
            }
          }
        }
      };
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Rank Preqs
      m_preqs[m_levels["rank"]][m_commands["REFab"]] = Lambdas::Preq::Rank::RequireAllBanksClosed<LPDDR5PIM>;
      m_preqs[m_levels["rank"]][m_commands["RFMab"]] = Lambdas::Preq::Rank::RequireAllBanksClosed<LPDDR5PIM>;

      m_preqs[m_levels["rank"]][m_commands["REFpb"]] = [this] (Node* node, int cmd, int target_id, Clk_t clk) {
        int another_target_bank_id = target_id + 8;

        for (auto bg : node->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            int num_banks_per_bg = m_organization.count[m_levels["bank"]];
            int flat_bankid = bank->m_node_id + bg->m_node_id * num_banks_per_bg;
            if (flat_bankid == target_id || flat_bankid == another_target_bank_id) {
              switch (bank->m_state) {
                case m_states["Pre-Opened"]: return m_commands["PRE"];
                case m_states["Opened"]: return m_commands["PRE"];
              }
            }
          }
        }

        return cmd;
      };
      
      m_preqs[m_levels["rank"]][m_commands["RFMpb"]] = m_preqs[m_levels["rank"]][m_commands["REFpb"]];

      // Bank Preqs
      m_preqs[m_levels["bank"]][m_commands["RD16"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        switch (node->m_state) {
          case m_states["Closed"]: return m_commands["ACT-1"];
          case m_states["Pre-Opened"]: return m_commands["ACT-2"];
          case m_states["Opened"]: {
            if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
              Node* rank = node->m_parent_node->m_parent_node;
              if (rank->m_final_synced_cycle < clk) {
                return m_commands["CASRD"];
              } else {
                return cmd;
              }
            } else {
              return m_commands["PRE"];
            }
          }    
          default: {
            spdlog::error("[Preq::Bank] Invalid bank state for an RD/WR command!");
            std::exit(-1);      
          } 
        }
      };
      m_preqs[m_levels["bank"]][m_commands["WR16"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        switch (node->m_state) {
          case m_states["Closed"]: return m_commands["ACT-1"];
          case m_states["Pre-Opened"]: return m_commands["ACT-2"];
          case m_states["Opened"]: {
            if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
              Node* rank = node->m_parent_node->m_parent_node;
              if (rank->m_final_synced_cycle < clk) {
                return m_commands["CASWR"];
              } else {
                return cmd;
              }
            } else {
              return m_commands["PRE"];
            }
          }    
          default: {
            spdlog::error("[Preq::Bank] Invalid bank state for an RD/WR command!");
            std::exit(-1);      
          } 
        }
      };

      // PIM bank preqs
      m_preqs[m_levels["bank"]][m_commands["MACAB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        Node* rank = node->m_parent_node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            switch (bank->m_state) {
              case m_states["Closed"]: return m_commands["ACTAB"];
              case m_states["Pre-Opened"]: return m_commands["PREA"];
              case m_states["Opened"]: {
                if (bank->m_row_state.find(target_id) == bank->m_row_state.end()) {
                  return m_commands["PREA"];
                }
                break;
              }
              default: {
                spdlog::error("[Preq::Bank] Invalid bank state for MACAB.");
                std::exit(-1);
              }
            }
          }
        }
        return cmd;
      };

      m_preqs[m_levels["bank"]][m_commands["MACSB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        Node* rank = node->m_parent_node->m_parent_node;
        for (auto bg : rank->m_child_nodes) {
          for (auto bank : bg->m_child_nodes) {
            if (bank->m_node_id != node->m_node_id) {
              continue;
            }
            switch (bank->m_state) {
              case m_states["Closed"]: return m_commands["ACTSB"];
              case m_states["Pre-Opened"]: return m_commands["PRESB"];
              case m_states["Opened"]: {
                if (bank->m_row_state.find(target_id) == bank->m_row_state.end()) {
                  return m_commands["PRESB"];
                }
                break;
              }
              default: {
                spdlog::error("[Preq::Bank] Invalid bank state for MACSB.");
                std::exit(-1);
              }
            }
          }
        }
        return cmd;
      };

      m_preqs[m_levels["bank"]][m_commands["MACPB"]] = [] (Node* node, int cmd, int target_id, Clk_t clk) {
        switch (node->m_state) {
          case m_states["Closed"]: return m_commands["ACTPB"];
          case m_states["Pre-Opened"]: return m_commands["ACT-2"];
          case m_states["Opened"]:
            if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
              return cmd;
            } else {
              return m_commands["PREPB"];
            }
          default: {
            spdlog::error("[Preq::Bank] Invalid bank state for MACPB.");
            std::exit(-1);
          }
        }
      };
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["bank"]][m_commands["RD16"]]  = Lambdas::RowHit::Bank::RDWR<LPDDR5PIM>;
      m_rowhits[m_levels["bank"]][m_commands["WR16"]]  = Lambdas::RowHit::Bank::RDWR<LPDDR5PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowHit::Bank::RDWR<LPDDR5PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowHit::Bank::RDWR<LPDDR5PIM>;
      m_rowhits[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowHit::Bank::RDWR<LPDDR5PIM>;
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["bank"]][m_commands["RD16"]]  = Lambdas::RowOpen::Bank::RDWR<LPDDR5PIM>;
      m_rowopens[m_levels["bank"]][m_commands["WR16"]]  = Lambdas::RowOpen::Bank::RDWR<LPDDR5PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACAB"]] = Lambdas::RowOpen::Bank::RDWR<LPDDR5PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACSB"]] = Lambdas::RowOpen::Bank::RDWR<LPDDR5PIM>;
      m_rowopens[m_levels["bank"]][m_commands["MACPB"]] = Lambdas::RowOpen::Bank::RDWR<LPDDR5PIM>;
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
      }
    };
};


}        // namespace Ramulator
