Compact LLM input for 180 URLLC + 180 eMBB tasks

User groups per service:
1-60:   G1,G2,W1,W2,W3,W4
61-120: S1,S2
121-180:G1,G2,W1,W2,W3,W4,S1,S2

Files:
E_TASK_180.csv
  One column: eMBB input data size in bits.

U_TASK_180.csv
  Two columns: URLLC input data size in bits, hard deadline in ms.

E_UL_GAIN_DB_180.csv
U_UL_GAIN_DB_180.csv
  Candidate uplink channel power gains only.
  Values are 10*log10(|h|^2), rounded to 1 decimal place.
  Row number equals the service-local user index.

SAT_DOWN_GAIN_DB.csv
  One row with two values:
  S1-to-cloud, S2-to-cloud channel power gains in dB.
  These global downlink gains are stored only once.

CPU cycles are omitted because beta = 41.25 * input_data_size_bits.
