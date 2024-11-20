PX4 Ulog file -> csv file including actuator output data
Then, sync recorded audio with actuator output data

1. px4_log.py exports actuator output data from the flight log file to the csv file.
   ```
   python px4_log.py
   ```

2. px4_log.py helps to sync the recorded audio with the motor output data.
   ```
   python audio_sync.py
   ```
   
