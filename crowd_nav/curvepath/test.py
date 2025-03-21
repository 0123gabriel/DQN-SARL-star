import re

# Example log line
log_file = '6/6 episode: Success! Total Reward: 11.556, R_dan: 0.128, 2*R_stop: 0.0, R_for: 0.76 R_goal: 10.0, R_col: 0.0, R_km: 0.7, R_col_wall: 0.0, timeout: 0.0, R_wall_min_dist: 0.128, R_wall_max_dist: 0.0, R_end: 0.0, R_stop_t: -0.16'

pattern = r'6/6 episode: Success! Total Reward: 11.556, R_dan: 0.128, 2*R_stop: 0.0, R_for: 0.76 R_goal: 10.0, R_col: 0.0, R_km: 0.7, R_col_wall: 0.0, timeout: 0.0, R_wall_min_dist: 0.128, R_wall_max_dist: 0.0, R_end: 0.0, R_stop_t: -0.16'

pattern = r' ([^:]+): (\d+\.\d+)' 

numbers = re.findall(pattern, log_file)

print("Extracted numbers:", numbers)

if 'abc':
    print('yes')


