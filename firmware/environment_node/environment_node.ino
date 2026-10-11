#include <WiFi.h>
#include <WiFiUdp.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ==========================================
// CONFIGURATION
// ==========================================
const char* ssid = "$$";
const char* password = "00000000";

// PLACEHOLDER: Replace with actual IP from ipconfig
const char* LAPTOP_IP = "192.168.1.100"; 
const int UDP_PORT = 5005;

#define DHTPIN 4
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);
WiFiUDP udp;

unsigned long lastSendTime = 0;
const int SEND_INTERVAL = 1000; // ~1 Hz

void setup() {
  Serial.begin(115200);
  delay(10);
  
  Serial.println("\nTHERMORA Environment Node Starting...");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("Node IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Target Laptop IP: ");
  Serial.println(LAPTOP_IP);

  dht.begin();
}

void loop() {
  unsigned long currentTime = millis();
  
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    
    if (isnan(h) || isnan(t)) {
      Serial.println("Failed to read from DHT sensor!");
    } else {
      StaticJsonDocument<128> doc;
      doc["device"] = "ENV_01";
      doc["type"] = "environment";
      doc["timestamp_ms"] = currentTime;
      doc["temperature_c"] = t;
      doc["humidity_pct"] = h;
      
      udp.beginPacket(LAPTOP_IP, UDP_PORT);
      serializeJson(doc, udp);
      udp.endPacket();
      
      Serial.print("Sent: ");
      Serial.print(t);
      Serial.print(" C, ");
      Serial.print(h);
      Serial.println(" %");
    }
    
    lastSendTime = currentTime;
  }
}
