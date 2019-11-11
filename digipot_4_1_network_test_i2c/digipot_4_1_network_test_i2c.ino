#include <SPI.h>
#include <i2c_t3.h>

// Comment out this #define to disable debugging output
#define DEBUG 1

// I2C macros

#define SLAVE_ADDRESS 0x04

// Digital pot macros

#define SPI_CLOCK_RATE 4000000  // MCP40210 max clock rate for daisy-chain configuration is 5.8 MHz
#define WRITE_COMMAND 0x10
#define P0_WRITE_COMMAND WRITE_COMMAND | 0x01
#define P1_WRITE_COMMAND WRITE_COMMAND | 0x02

// Pin selection macros

#define CHIP_SELECT_PIN 9

typedef struct NeuronLayer {
   int inputsPerNeuron;
   int neuronCount;
   byte **weights;
} NeuronLayer;

typedef struct NeuralNetwork {
   int layerCount;
   NeuronLayer **layers;
   uint8_t *weights;
   int weightCount;
   int neuronCount;
} NeuralNetwork;

// Output read pins
const byte _outputPins[5] = {33,34,35,36,37};

uint16_t outputs[5] = {1,2,3,4,5};
byte weights[20];

// Set up MLP layers.  
byte *l0w = weights;
byte *layer_0_weights_by_neuron[] = {l0w, l0w+4, l0w+8, l0w+12};
NeuronLayer layer0 = { .inputsPerNeuron = 4, .neuronCount = 4, .weights = layer_0_weights_by_neuron };

byte *l1w = l0w + (layer0.inputsPerNeuron * layer0.neuronCount);
byte *layer_1_weights_by_neuron[] = {l1w }; //, l1w+4, l1w+8, l1w+12};
NeuronLayer layer1 = { .inputsPerNeuron = 4, .neuronCount = 1, .weights = layer_1_weights_by_neuron };

NeuronLayer *layers[] = { &layer0, &layer1 };
NeuralNetwork mlp = { .layerCount = 2, .layers = layers, .weights = weights, .weightCount = 20, .neuronCount = 5 };
NeuralNetwork *MLP = &mlp;

const byte _P0WriteCommand = P0_WRITE_COMMAND;
const byte _P1WriteCommand = P1_WRITE_COMMAND;
const byte _default_weight = 0;

void setup() 
{
  char serialDebugBuff[100];
  int buttonState = 0;
  int layerIndex = 0;
  int neuronIndex = 0;
  int synapseIndex = -1;
 
  
  for (int i; i < MLP->weightCount; i++)
    weights[i] = _default_weight;
  
  SPI.begin();
  pinMode(CHIP_SELECT_PIN, OUTPUT);
  digitalWrite(CHIP_SELECT_PIN, HIGH);

#ifdef DEBUG
  Serial.begin(9600); // start serial for output
#endif
  
  // initialize i2c as slave
  Wire.begin(SLAVE_ADDRESS);
  Wire.setClock(400000);
  
  // define callbacks for i2c communication
  Wire.onReceive(receiveCommand);
  Wire.onRequest(sendData);

  analogReadResolution(13);
}

void loop()
{
//  delay(1);
}

void readOutputs()
{
  const int read_count = 2;

#ifdef DEBUG
  Serial.println("Reading outputs:");
  Serial.flush();
#endif
    
  for (int i = 0; i < MLP->neuronCount; i++)
  {
    uint16_t read = 0;
    for (int j = 0; j < read_count; j++)
      read += analogRead(_outputPins[i]);
      
    outputs[i] = read >> 1;
  }
}

void receiveCommand(int byteCount)
{  
  Serial.print("Bytes received: ");
  Serial.print(byteCount);
  Serial.println("");
  Serial.flush();
  
  if(Wire.available())
  {
    int command = Wire.read();

#ifdef DEBUG
    Serial.print("Command received: ");
    Serial.print(command);
#endif

    if (command == 0)
    {
      receiveData(byteCount-1);
    }
  }

#ifdef DEBUG
  while(Wire.available())
  {
    Serial.print(" ");
    Serial.print(Wire.read());
  }

  Serial.println("");
#endif
}

// callback for received data
void receiveData(int byteCount)
{
  Serial.println("");
  
  int weightIdx = 0;
  int weightCount = MLP->weightCount;

  char tempweights[20];

  Wire.read((char*)MLP->weights, 20);

#ifdef DEBUG
  Serial.print("Weights received: [ ");
  for (int i = 0; i < weightCount; i++)
  {
    Serial.print(MLP->weights[i]);
    Serial.print( i < (weightCount-1) ? ", " : " ]");
  }
  Serial.println("");
#endif

  setDigitalPots(MLP->weights, MLP->weightCount);
}

// callback for sending data
void sendData()
{
  readOutputs();
  
  char *bytes = (char*)outputs;
  int outputCount = 2 * sizeof(outputs)/sizeof(outputs[0]);

#ifdef DEBUG
  Serial.print("Sending output bytes: [ ");
  for (int i = 0; i < outputCount; i++)
  {
    Serial.print((int)(bytes[i]));
    Serial.print( i < (outputCount-1) ? ", " : " ]");
  }
  Serial.println("");
#endif
      
  Wire.write((char*)outputs, outputCount);
}

void setDigitalPots(byte data[], int length)
{
  // Write all of the values for the P0 pots
  SPI.beginTransaction(SPISettings(SPI_CLOCK_RATE, MSBFIRST, SPI_MODE0));
  digitalWrite(CHIP_SELECT_PIN, LOW);
  for (int i = 0; i < length; i += 2)
  {
    SPI.transfer(_P0WriteCommand);
    SPI.transfer(data[i]);
  }
  digitalWrite(CHIP_SELECT_PIN, HIGH);
  SPI.endTransaction();

//   Write all of the values for the P1 pots
  SPI.beginTransaction(SPISettings(SPI_CLOCK_RATE, MSBFIRST, SPI_MODE0));
  digitalWrite(CHIP_SELECT_PIN, LOW);
  for (int j = 1; j < length; j += 2)
  {
    SPI.transfer(_P1WriteCommand);
    SPI.transfer(data[j]);
  }
  digitalWrite(CHIP_SELECT_PIN, HIGH);
  SPI.endTransaction();

  delayMicroseconds(10);
}
