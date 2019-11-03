#include <Wire.h>

#define SLAVE_ADDRESS 0x04
int number = 0;
int state = 0;

int weightIdx = 0;
int outputIdx = 0;

int weights[10];
int outputs[5];
int weightCount = sizeof(weights)/sizeof(weights[0]);

void(*rxCallback)(int) = nullptr;

void setup() 
{
   Serial.begin(9600); // start serial for output
  
   
   // initialize i2c as slave
   Wire.begin(SLAVE_ADDRESS);
   
   // define callbacks for i2c communication
   rxCallback = &receiveCommand;
   Wire.onReceive(rxCallback);
   Wire.onRequest(sendData);

   Serial.println("Ready!");
}

void loop() 
{
   delay(1);
}

void receiveCommand(int byteCount)
{
  if(Wire.available())
  {
    int command = Wire.read();

    Serial.print("Command received: ");
    Serial.println(command);

    if (command == 0)
    {
      rxCallback = &receiveData;
    }
    else
    {
      outputIdx = 0;
    }
  }
}

// callback for received data
void receiveData(int byteCount)
{
  if (Wire.available())
  {
    weights[weightIdx++] = Wire.read();

    if (weightIdx >= weightCount)
    {
      weightIdx = 0;
      rxCallback = &receiveCommand;
      
      Serial.print("Weights received: [ ");
      for (int i = 0; i < weightCount; i++)
      {
        Serial.print(weights[i]);
        Serial.print( i < (weightCount-1) ? ", " : " ]");
      }
      Serial.println("");
    }
  }
}

// callback for sending data
void sendData()
{
   Wire.write(outputs[outputIdx++]);
}
