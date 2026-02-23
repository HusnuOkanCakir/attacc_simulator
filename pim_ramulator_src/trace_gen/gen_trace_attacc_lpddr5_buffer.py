import argparse
import math

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 1
n_rank = 1
n_bank = 4
n_bg = 4
n_row = pow(2, 13)  # LPDDR5_2Gb_x16 row count
n_col = pow(2, 7)  # Effective LPDDR5 column count at 32B access granularity
prefetch_size = 32 # byte
n_mac = 16


# Granularity size
LPDDR_GS = {}
LPDDR_GS['col']     = prefetch_size
LPDDR_GS['row']     = n_col * LPDDR_GS['col']
LPDDR_GS['ba']      = n_row * LPDDR_GS['row']
LPDDR_GS['bg']      = n_bank * LPDDR_GS['ba']
LPDDR_GS['rank']    = n_bg * LPDDR_GS['bg']
LPDDR_GS['pch']     = n_rank * LPDDR_GS['rank']
LPDDR_GS['ch']      = n_pch * LPDDR_GS['pch']
LPDDR_GS['hbm']     = n_channel * LPDDR_GS['ch']
LPDDR_GS['attacc']  = max_n_hbm * LPDDR_GS['hbm']


## ------------------------------------  LPDDR5 memory space ----------------------------------------##
## ---| legacy CH | rank | BG | BA | row index | effective col index | access granularity |---------------------##
## bits |    4    |  0   | 2  | 2  |    15     |          7          |         5          |                     ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []

def cmd_list_reset():
  cmd_score_wrgb   = []
  cmd_score_mac    = []
  cmd_score_mvsb   = []
  cmd_sfm          = []
  cmd_context_mvgb = []
  cmd_context_mac  = []
  cmd_context_mvsb = []

  valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel):
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_sfm.append([])
  cmd_context_mvgb.append([])
  cmd_context_mac.append([])
  cmd_context_mvsb.append([])

  valid_channels.append(valid_channel);

  def score_cpvec(addr_offset, L):
    ## (pCH) C R (MAC)
    ## write input vector to gemv buffer
    # number of partition = (R parallel units)

    for col_idx in range(math.ceil(dhead / n_mac)):
      for lch in range(math.ceil(valid_channel)):
        # GEMV buffer address, col granularity = 1
        addr = addr_offset + lch * LPDDR_GS['ch'] + col_idx
        hex_addr = hex(addr)[2:]
        cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

  def score_mac(addr_offset, L):
    ## (pCH) C R (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1 x k) x Matrix (k x n) multiplication
    ## GEMV unit = adder tree mode
    for n_idx in range(math.ceil(L / n_pch)):
      cmd_score_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / n_mac)):
        idx = k_idx + n_idx * math.ceil(dhead / n_mac) 

        bg_idx = idx % (n_bg * n_rank)
        num_bg_indices = int(idx / (n_bg * n_rank))

        bank_idx = num_bg_indices % (n_bank)
        num_bank_indices = int(num_bg_indices / n_bank)

        col_idx = num_bank_indices % (int(LPDDR_GS['row'] / LPDDR_GS['col']))
        row_idx = int(num_bank_indices / (int(LPDDR_GS['row'] / LPDDR_GS['col'])))

        # All bank command (legacy channel)
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * LPDDR_GS['ch'] + bg_idx * LPDDR_GS['bg'] + \
                 bank_idx * LPDDR_GS['ba'] + row_idx * LPDDR_GS['row'] + col_idx * LPDDR_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_score_mac[itr][-1].append("PIM_MAC_PB 0x{0:0>8}".format(hex_addr))
         ## parallelization

      ## MVSB command (Move to Softmax buffer) 
      ## A output element is generated for every n_idx
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch) - 1:
        cmd_score_mvsb[itr].append([])
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * LPDDR_GS['ch']
          hex_addr = hex(addr)[2:]
          cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def context_cpvec(addr_offset, L):
    ## (pCH) R C (MAC)
    ## write input vector to gemv buffer
    # number of columns of partition = L / (R parallel units)
    for col_idx in range(math.ceil(L / (n_pch * n_mac))):
      for lch in range(math.ceil(valid_channel)):
        # GEMV buffer address, col granularity = 1
        addr = addr_offset + lch * LPDDR_GS['ch'] + col_idx
        hex_addr = hex(addr)[2:]
        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

  def context_mac(addr_offset, L):
    ## (pCH) R C (MAC)
    # MAC and move output vector to softmax buffer
    ## Vector (1xk) x Matrix (k x n ) multiplication
    ## GEMV unit = mac mode
    for n_idx in range(math.ceil(dhead / (n_mac))):
      cmd_context_mac[itr].append([])
      for k_idx in range(math.ceil(L / (n_pch))):
        idx = k_idx + n_idx * math.ceil(L / (n_pch))
        bg_idx = idx % (n_bg * n_rank) 
        num_bg_indices = int(idx / (n_bg * n_rank))

        bank_idx = num_bg_indices % (n_bank)
        num_bank_indices = int(num_bg_indices / n_bank)

        col_idx = num_bank_indices % (int(LPDDR_GS['row'] / LPDDR_GS['col']))
        row_idx = int(num_bank_indices / (int(LPDDR_GS['row'] / LPDDR_GS['col'])))

        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * LPDDR_GS['ch'] + bg_idx * LPDDR_GS['bg'] + \
                 bank_idx * LPDDR_GS['ba'] + row_idx * LPDDR_GS['row'] + col_idx * LPDDR_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_context_mac[itr][-1].append("PIM_MAC_PB 0x{0:0>8}".format(hex_addr))

      ## parallelization. Generate 16 elements per n_idx
      cmd_context_mvsb[itr].append([])
      for lch in range(math.ceil(valid_channel)):
        addr = addr_offset + lch * LPDDR_GS['ch']
        hex_addr = hex(addr)[2:]
        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def softmax(L):
    for lch in range(math.ceil(valid_channel)):
      addr = lch * LPDDR_GS['ch']
      hex_addr = hex(addr)[2:]
      cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

  score_cpvec(key_addr, L)

  score_mac(key_addr, L)

  softmax(L)

  context_cpvec(val_addr, L)

  context_mac(val_addr, L)


def run_attention(dhead, n_head_per_hbm, L, trace_file_name):
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
  head_offset = partition_size
  v_offset = pow(2, 23) 
  

  cmd_list_reset()
  ##-- Generate Commands --##
  num_itr = math.ceil(n_head_per_hbm / (n_channel))
  for itr in range(num_itr):
    remainder = 0
    if (n_head_per_hbm / ((itr+1) * n_channel) < 1):
      remainder = n_head_per_hbm % n_channel
    key_addr = itr * partition_size 
    val_addr = key_addr + v_offset
    if remainder == 0:
      Attention(L, key_addr, val_addr, itr)
    else:
      Attention(L, key_addr, val_addr, itr, remainder)


  ##-- Ovelapping Commands --##
  barrier = []
  for lch in range(n_channel):
    addr = lch * LPDDR_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  total_cmd = []
  for i in range(0, num_itr -1, 2):

    # Head0: Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## dummy MAC
    if i == 0:
      for j in range(valid_channels[i]):
        total_cmd.append(cmd_score_mac[i][0][j])
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## WRGB (Head1)
      if not j == length:
        stride = int(math.ceil(dhead/n_mac)*math.ceil(valid_channels[i+1])/length);
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
            break;
          total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: SoftMax, Head1: Score
    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC (Head1)
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i+1]):
            break;
          total_cmd += cmd_score_mac[i+1][j*stride+k]
      ## MVSB (Head1)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i+1][j-1]
      ## SFM (Head0)
      if j == 0:
        total_cmd += cmd_sfm[i]
      ## MVGB (Head0)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(math.ceil(L/(n_pch*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
              break;
            total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: Context, Head1: Softmax
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## SFM (Head1)
      if j == 0:
        total_cmd += cmd_sfm[i+1]
      ## MVGB (Head1)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(math.ceil(L/(n_pch*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
              break;
            total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head1: Context
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  if num_itr % 2 != 0:
    i = num_itr - 1

    # Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/16)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        stride = 16;
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break;
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # SoftMax
    ## SFM (Head0)
    total_cmd += cmd_sfm[i]
    ## MVGB (Head0)
    total_cmd += cmd_context_mvgb[i]
    ## BARRIER
    total_cmd += barrier

    # Context
    length = math.ceil(dhead/n_mac)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  trace_file = open(trace_file_name, 'w')
  for cmd in total_cmd:
    trace_file.write(cmd + "\n")

  trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac


  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=64,
                      help="Number of heads, default=64")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_buffer.trace", 
                      help="output path")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(LPDDR_GS['col'] / data_size)

  print("------   Make a trace of buffer-level AttAcc (LPDDR5)   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  run_attention(dhead, n_head_per_hbm, L, args.output)



if __name__ == "__main__":
  main()
