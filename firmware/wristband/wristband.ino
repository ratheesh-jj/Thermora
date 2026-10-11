#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <MAX30105.h>
#include <ArduinoJson.h>

// ==========================================
// CONFIGURATION
// ==========================================
const char* ssid = "$$";
const char* password = "00000000";

// PLACEHOLDER: Replace with actual IP from ipconfig
const char* LAPTOP_IP = "192.168.1.100"; 
const int UDP_PORT = 5005;

WiFiUDP udp;

Adafruit_MPU6050 mpu;
MAX30105 particleSensor;

unsigned long lastMpuTime = 0;
const int MPU_INTERVAL = 50; // ~20 Hz

unsigned long lastStatusTime = 0;
int packetCount = 0;

void setup() {
  Serial.begin(115200);
  delay(10);

  Serial.println("\nTHERMORA Wristband Node Starting...");

  // WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("Wristband IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Target Laptop IP: ");
  Serial.println(LAPTOP_IP);

  Wire.begin(21, 22);

  // MPU6050 setup
  if (!mpu.begin(0x68)) {
    Serial.println("Failed to find MPU6050 chip");
  } else {
    Serial.println("MPU6050 OK");
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }

  // MAX30102 setup
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) { // I2C speed fast (400kHz) is crucial for 100Hz
    Serial.println("MAX30102 was not found. Please check wiring/power.");
  } else {
    Serial.println("MAX30102 OK");
    byte ledBrightness = 60; // Options: 0=Off to 255=50mA
    byte sampleAverage = 4; // Options: 1, 2, 4, 8, 16, 32
    byte ledMode = 2; // Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
    int sampleRate = 100; // Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
    int pulseWidth = 411; // Options: 69, 118, 215, 411
    int adcRange = 4096; // Options: 2048, 4096, 8192, 16384

    particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); 
  }
}

void loop() {
  unsigned long currentTime = millis();
  
  // MAX30102 streams as fast as data is ready (target 100Hz)
  particleSensor.check();
  
  while (particleSensor.available()) {
    uint32_t ir = particleSensor.getFIFOIR();
    uint32_t red = particleSensor.getFIFORed();
    particleSensor.nextSample();
    
    // We send a packet per PPG sample, appending the latest IMU data if it exists.
    // To minimize bandwidth and formatting time, we can send compact JSON.
    StaticJsonDocument<256> doc;
    doc["device"] = "WRISTBAND_01";
    doc["type"] = "wristband";
    doc["timestamp_ms"] = millis();
    
    JsonObject ppg = doc.createNestedObject("ppg");
    ppg["ir"] = ir;
    ppg["red"] = red;
    
    // Only add IMU at 20Hz
    if (currentTime - lastMpuTime >= MPU_INTERVAL) {
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      
      JsonObject imu = doc.createNestedObject("imu");
      imu["ax"] = a.acceleration.x / 9.81; // Convert to g
      imu["ay"] = a.acceleration.y / 9.81;
      imu["az"] = a.acceleration.z / 9.81;
      imu["gx"] = g.gyro.x * 57.2958; // Convert to deg/s
      imu["gy"] = g.gyro.y * 57.2958;
      imu["gz"] = g.gyro.z * 57.2958;
      
      lastMpuTime = currentTime;
    }
    
    udp.beginPacket(LAPTOP_IP, UDP_PORT);
    serializeJson(doc, udp);
    udp.endPacket();
    
    packetCount++;
  }
  
  // Status print every 5 seconds
  if (currentTime - lastStatusTime >= 5000) {
    Serial.print("Status OK. Packets/sec: ");
    Serial.println(packetCount / 5);
    packetCount = 0;
    lastStatusTime = currentTime;
  }
}
