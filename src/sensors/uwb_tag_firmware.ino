/*
  UWB Tag firmware (template)
  ------------------------------------------------
  Flash this onto the ONE UWB module worn on the wrist, alongside the
  Core2 IMU. It continuously ranges against both anchors and streams
  each measurement as a CSV line over USB serial:

      timestamp_ms,anchor_address,range_m

  You'll get interleaved lines from both anchors -- the Python
  collector groups them by anchor address into two aligned columns
  (range_anchor1, range_anchor2) during preprocessing.

  Same library/pin caveats as uwb_anchor_firmware.ino apply here --
  confirm against your actual UWB module before flashing.
*/

#include <SPI.h>
#include <DW1000Ranging.h>

const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS  = 4;

void setup() {
  Serial.begin(115200);
  delay(1000);

  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  // Start as the mobile tag. It will automatically discover and range
  // against both anchors that are running uwb_anchor_firmware.ino.
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C",
                            DW1000.MODE_LONGDATA_RANGE_LOWPOWER);

  Serial.println("timestamp_ms,anchor_address,range_m");
}

void loop() {
  DW1000Ranging.loop();
}

void newRange() {
  unsigned long now = millis();
  Serial.print(now);
  Serial.print(",");
  Serial.print(DW1000Ranging.getDistantDevice()->getShortAddress(), HEX);
  Serial.print(",");
  Serial.println(DW1000Ranging.getDistantDevice()->getRange());
}

void newDevice(DW1000Device* device) {
  // Discovered an anchor -- nothing to log, ranging starts automatically.
}

void inactiveDevice(DW1000Device* device) {
  // Lost contact with an anchor -- if this happens a lot during
  // collection, move the anchor closer or check line-of-sight.
}
