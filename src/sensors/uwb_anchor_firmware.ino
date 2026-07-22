/*
  UWB Anchor firmware (template)
  ------------------------------------------------
  Flash this onto TWO of your three UWB modules. Each acts as a fixed
  "anchor" at a known point around your gesture space (e.g. one placed
  directly in front of where the person stands, one placed off to the
  side). The third module (see uwb_tag_firmware.ino) is worn on the
  wrist and continuously measures its range to each anchor.

  NOTE: This uses the popular thotro/arduino-dw1000 library and its
  DW1000Ranging helper, which is the standard approach for DW1000-based
  UWB boards (e.g. ESP32 + DW1000 modules). If your kit's UWB modules
  use a different chip (e.g. DW3000 / Qorvo DWM3000, or a DWM1001-DEV
  running the pre-flashed PANS firmware with a UART shell), the pin
  setup and library calls below will differ -- check the label/model
  printed on your board and swap in the matching library. Ask the TA
  if you're not sure which module you have.

  Install via Library Manager: "DW1000" (thotro/arduino-dw1000) or
  the vendor-specific library that matches your board.
*/

#include <SPI.h>
#include <DW1000Ranging.h>

// TODO: set a UNIQUE short address for each anchor, e.g. "82:17:5B:D5:A9:9A:E2:9C"
// Give anchor #1 and anchor #2 different addresses.
#define ANCHOR_ADD "82:17:5B:D5:A9:9A:E2:9C"

// TODO: confirm these pins match your specific ESP32 + DW1000 wiring
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

  // Start as an anchor. The tag will initiate ranging with each anchor.
  DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_LONGDATA_RANGE_LOWPOWER);

  Serial.print("Anchor ready: ");
  Serial.println(ANCHOR_ADD);
}

void loop() {
  DW1000Ranging.loop();
}

void newRange() {
  // The anchor itself doesn't need to log anything -- the tag records
  // and streams the range values. This callback exists so the library
  // stays happy; leave it empty or add a debug print if useful.
}

void newDevice(DW1000Device* device) {
  Serial.print("New ranging device: ");
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device* device) {
  Serial.print("Device went inactive: ");
  Serial.println(device->getShortAddress(), HEX);
}
