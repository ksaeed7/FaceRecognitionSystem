#include "WiFi.h"
#include "esp_camera.h"

// Replace with your network credentials
const char* ssid = "username";
const char* password = "password";

// The IP address and port of the server you want to connect to
const char* serverIP = "ipaddress";
const uint16_t serverPort = port;

const int buttonPin = 0;   // GPIO0 for the button
const int ledPin = 4;      // GPIO2 for the built-in LED (or use another pin for an external LED)

WiFiClient client;

void setup() {
  Serial.begin(115200);
  Serial.println("Starting Wifi...");
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Configure the camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  
  pinMode(ledPin, OUTPUT);           // Initialize the LED pin as an output


  while(1)
  {
   if (!client.connect(serverIP, serverPort)) {
    Serial.println("Failed to connect to server");
    delay(500);
  }
  else{
    break;
  }
  }
}
uint16_t count = 0;
bool isOn = false;
void loop() {
  // Capture a photo
  if (false){//count % 10000 == 0) {
        isOn = !isOn;
        // Button is pressed, turn on the LED
        if (isOn)
        {
        digitalWrite(ledPin, HIGH);
        }
        else {
            // Button is not pressed, turn off the LED
            digitalWrite(ledPin, LOW);
        }
        count = 0;
  }

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  // Check if the client is still connected
  if (!client.connected()) {
    // Attempt to reconnect
    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Failed to reconnect to server");
      // Handle reconnection failure
      esp_camera_fb_return(fb);
    }
  }
  else{

  // Send image size and data
  uint32_t imageSize = fb->len;
  client.write((char *)&imageSize, sizeof(imageSize));
  client.write((char *)fb->buf, fb->len);
  Serial.println(imageSize);

  // Return the frame buffer
  esp_camera_fb_return(fb);
  }
  //count += 1;
  // Add a delay if necessary, or perform other tasks
}

