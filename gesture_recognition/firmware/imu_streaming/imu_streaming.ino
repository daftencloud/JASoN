/*
  Minimal IMU streaming firmware -- M5Stack Core2
  ------------------------------------------------
  The ONE unavoidable piece of firmware in this project: something has
  to physically read the accelerometer/gyro chip and print it over
  serial, since Python runs on your laptop, not on the ESP32. This
  file does ONLY that -- no gesture logic, no classification, no
  feature extraction. All of that lives in Python (src/sensors/imu_reader.py
  and src/extract_features.py) per the project's pure-Python design.

  Streams CSV lines matching what imu_reader.py expects:
      timestamp_ms,ax,ay,az,gx,gy,gz

  Requires: M5Unified, M5Core2 libraries (Arduino Library Manager).
*/

#include <M5Unified.h>

const int SAMPLE_RATE_HZ = 100;
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
unsigned long lastSampleTime = 0;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  Serial.begin(115200);
  delay(500);
  Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  M5.update();

  float ax, ay, az, gx, gy, gz;
  M5.Imu.getAccel(&ax, &ay, &az);
  M5.Imu.getGyro(&gx, &gy, &gz);

  Serial.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n",
                now, ax, ay, az, gx, gy, gz);
}
