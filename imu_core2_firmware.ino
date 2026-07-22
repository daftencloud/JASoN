/*
  IMU streaming firmware for M5Stack Core2
  ------------------------------------------------
  Reads the onboard IMU (accelerometer + gyroscope) and streams
  each sample as a CSV line over USB serial:

      timestamp_ms,ax,ay,az,gx,gy,gz

  timestamp_ms is milliseconds since boot (millis()).
  ax/ay/az are in g's, gx/gy/gz are in deg/s.

  Uses the M5Unified / M5Core2 library, which wraps the onboard
  MPU6886 (or BMI270 on newer boards) IMU. Install "M5Unified" and
  "M5Core2" via the Arduino Library Manager if not already present.
*/

#include <M5Unified.h>

// How often to sample, in Hz. 50-100 Hz is typical for arm gestures.
const int SAMPLE_RATE_HZ = 100;
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;

unsigned long last_sample_time = 0;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);

  Serial.begin(115200);
  delay(500);

  // Print a CSV header once so the Python collector can sanity-check
  // the port before it starts logging.
  Serial.println("timestamp_ms,ax,ay,az,gx,gy,gz");
}

void loop() {
  unsigned long now = millis();
  if (now - last_sample_time < SAMPLE_INTERVAL_MS) {
    return;  // not time for the next sample yet
  }
  last_sample_time = now;

  M5.update();

  float ax, ay, az;
  float gx, gy, gz;

  M5.Imu.getAccel(&ax, &ay, &az);
  M5.Imu.getGyro(&gx, &gy, &gz);

  Serial.printf("%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f\n",
                now, ax, ay, az, gx, gy, gz);
}
