#include <SPI.h>
#include <i2c_t3.h>

// Comment out this #define to disable debugging output
#define DEBUG

#ifdef DEBUG
  #define DEBUG_PRINT(X) Serial.print(X); Serial.flush()
  #define DEBUG_PRINTLN(X) Serial.println(X); Serial.flush()
#else
  #define DEBUG_PRINT(X)
  #define DEBUG_PRINTLN(X)
#endif

// Neural network dimensions
#define NEURON_COUNT 7
#define WEIGHT_COUNT 28
#define INPUT_LAYER_SIZE 4

// I2C macros

#define SLAVE_ADDRESS 0x04

#define INITIALIZE_CMD 0x01
#define SET_WEIGHTS_CMD 0x02
#define SET_INPUTS_CMD 0x03
#define READ_OUTPUTS_CMD 0x04

// Shift register input macros

#define INPUT_SPI SPI1
#define INPUT_LATCH_PIN 31

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

// Input driver pins
const byte _inputDriverPins[INPUT_LAYER_SIZE] = {25,26,27,28};

// Output read pins
const byte _outputReadPins[NEURON_COUNT] = {33,34,35,36,37,38,39};

byte inputs[INPUT_LAYER_SIZE];
uint16_t outputs[NEURON_COUNT] = {1,2,3,4,5,6,7};
byte weights[WEIGHT_COUNT];

// Set up MLP layers.  
byte *l0w = weights;
byte *layer_0_weights_by_neuron[] = {l0w, l0w+4, l0w+8, l0w+12};
NeuronLayer layer0 = { .inputsPerNeuron = 4, .neuronCount = 4, .weights = layer_0_weights_by_neuron };

byte *l1w = l0w + (layer0.inputsPerNeuron * layer0.neuronCount);
byte *layer_1_weights_by_neuron[] = {l1w, l1w+4, l1w+8};
NeuronLayer layer1 = { .inputsPerNeuron = 4, .neuronCount = 3, .weights = layer_1_weights_by_neuron };

NeuronLayer *layers[] = { &layer0, &layer1 };
NeuralNetwork mlp = { .layerCount = 2, .layers = layers, .weights = weights, .weightCount = WEIGHT_COUNT, .neuronCount = NEURON_COUNT };
NeuralNetwork *MLP = &mlp;

const byte _P0WriteCommand = P0_WRITE_COMMAND;
const byte _P1WriteCommand = P1_WRITE_COMMAND;
const byte _default_weight = 0;

SPISettings _digipotSPISettings = SPISettings(SPI_CLOCK_RATE, MSBFIRST, SPI_MODE0);
SPISettings _inputDriverSPISettings = SPISettings(SPI_CLOCK_RATE, LSBFIRST, SPI_MODE0);

void setup() 
{  
  for (int i; i < MLP->weightCount; i++)
    weights[i] = _default_weight;
    
  for (int j; j < INPUT_LAYER_SIZE; j++)
  {
    pinMode(_inputDriverPins[j], OUTPUT);
    digitalWrite(_inputDriverPins[j], HIGH);
  }
  
  pinMode(CHIP_SELECT_PIN, OUTPUT);
  SPI.begin();

  pinMode(INPUT_LATCH_PIN, OUTPUT);
  INPUT_SPI.begin();

#ifdef DEBUG
  // Initialize serial communication for debugging output.
  Serial.begin(9600);
#endif
  
  // Initialize I2C communication.
  Wire.begin(SLAVE_ADDRESS);
  Wire.setClock(400000);
  
  Wire.onReceive(receiveCommand);
  Wire.onRequest(sendData);

  // Set ADC resolution.
  analogReadResolution(13);
}

void loop()
{
  
}

void readOutputs()
{
  const int read_count_power_of_2 = 3;
  const int read_count = 1 << read_count_power_of_2;

  DEBUG_PRINTLN("Reading outputs:");
    
  for (int i = 0; i < MLP->neuronCount; i++)
  {
    uint16_t read = 0;
    for (int j = 0; j < read_count; j++)
      read += analogRead(_outputReadPins[i]);
      
    outputs[i] = read >> read_count_power_of_2;
    
    DEBUG_PRINT(" ");
    DEBUG_PRINT(outputs[i]);
  }

  DEBUG_PRINTLN("");
}

void receiveCommand(int byteCount)
{  
  DEBUG_PRINT("Bytes received: ");
  DEBUG_PRINT(byteCount);
  DEBUG_PRINTLN("");
  
  if(Wire.available())
  {
    int command = Wire.read();

    DEBUG_PRINT("Command received: ");
    DEBUG_PRINT(command);

    if (command == SET_WEIGHTS_CMD)
    {
      receiveWeights(byteCount-1);
    }
    else if (command == SET_INPUTS_CMD)
    {
      receiveInputsForShiftRegister(byteCount-1);
    }
  }

#ifdef DEBUG
  while(Wire.available())
  {
    DEBUG_PRINT(" ");
    DEBUG_PRINT(Wire.read());
  }

  DEBUG_PRINTLN("");
#endif
}

// callback for received data
void receiveWeights(int byteCount)
{
  DEBUG_PRINTLN("");
  
  int weightIdx = 0;
  int weightCount = MLP->weightCount;

  char tempweights[WEIGHT_COUNT];

  Wire.read((char*)MLP->weights, WEIGHT_COUNT);

#ifdef DEBUG
  DEBUG_PRINT("Weights received: [ ");
  for (int i = 0; i < weightCount; i++)
  {
    DEBUG_PRINT(MLP->weights[i]);
    DEBUG_PRINT( i < (weightCount-1) ? ", " : " ]");
  }
  DEBUG_PRINTLN("");
#endif

  setDigitalPots(MLP->weights, MLP->weightCount);
}

void receiveInputsForShiftRegister(int byteCount)
{
  byte data;

  DEBUG_PRINTLN("");
  DEBUG_PRINT("Received ");
  DEBUG_PRINT(byteCount);
  DEBUG_PRINT(byteCount == 1 ? " byte. Sending input data: " : " bytes. Sending input data: ");
  

  for (int i = 0; i < byteCount && Wire.available(); i++)
  {
    data = Wire.read() ^ 0xFF;
    
    digitalWrite(INPUT_LATCH_PIN, LOW);
    INPUT_SPI.beginTransaction(SPISettings(SPI_CLOCK_RATE, LSBFIRST, SPI_MODE0));
    INPUT_SPI.transfer(data);
    INPUT_SPI.endTransaction();
    digitalWrite(INPUT_LATCH_PIN, HIGH);
    
#ifdef DEBUG
    byte mask = 0x80;
    for (int j = 0; j < 8; j++)
    {
      DEBUG_PRINT((data & mask) != 0 ? "1 " : "0 ");
      mask >>= 1;      
    }
#endif
  }

  //TODO Adjust or remove this delay once pulldown/pullup resistor values are finalized.
//  delayMicroseconds(100);
  
  DEBUG_PRINTLN("");

}

void receiveInputs(int byteCount)
{
  // Bitmask is set to 0x80 for reading input value from MSB.
  const byte bitMask = 0x80;
  int inputIdx = 0;

  DEBUG_PRINTLN("");
  DEBUG_PRINT("Input values received: ");

  // Unpack input values from bit fields.
  for (int i = 0; i <= INPUT_LAYER_SIZE/8; i++)
  {    
    byte data = Wire.read();

    for (int j = 0; j < 8 && inputIdx < INPUT_LAYER_SIZE; j++)
    {
      // Write input value to the appropriate pin.
//      digitalWrite(_inputDriverPins[inputIdx++], ((data & bitMask) == 0 ? LOW : HIGH));

      if ((data & bitMask) != 0)
      {
        pinMode(_inputDriverPins[inputIdx], OUTPUT);
        digitalWrite(_inputDriverPins[inputIdx++], HIGH);
      }
      else
      {
        // This should leave the pin in a floating state so that it can be pulled to 
        //   0V ground (Vss + 1.65V for the Teensy) through the pulldown resistor.
        digitalWrite(_inputDriverPins[inputIdx++], LOW);
        pinMode(_inputDriverPins[inputIdx++], INPUT);
      }

      DEBUG_PRINT((data & bitMask) == 0 ? LOW : HIGH);
      DEBUG_PRINT(" ");

      data <<= 1;
    }     
  }

  DEBUG_PRINTLN("");
}

void sendData()
{
  readOutputs();
  
  char *bytes = (char*)outputs;
  int outputCount = 2 * sizeof(outputs)/sizeof(outputs[0]);

#ifdef DEBUG
  DEBUG_PRINT("Sending output bytes: [ ");
  for (int i = 0; i < outputCount; i++)
  {
    DEBUG_PRINT((int)(bytes[i]));
    DEBUG_PRINT( i < (outputCount-1) ? ", " : " ]");
  }
  DEBUG_PRINTLN("");
#endif
      
  Wire.write((char*)outputs, outputCount);
}

void setDigitalPots(byte data[], int length)
{
  // Write all of the values for the P0 pots.
  SPI.beginTransaction(SPISettings(SPI_CLOCK_RATE, MSBFIRST, SPI_MODE0));
  digitalWrite(CHIP_SELECT_PIN, LOW);
  for (int i = 0; i < length; i += 2)
  {
    SPI.transfer(_P0WriteCommand);
    SPI.transfer(data[i]);
  }
  digitalWrite(CHIP_SELECT_PIN, HIGH);
  SPI.endTransaction();

  // Write all of the values for the P1 pots.
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
