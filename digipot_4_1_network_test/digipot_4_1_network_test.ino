#include <LiquidCrystal.h>
#include <SPI.h>

// Digital pot macros

#define SPI_CLOCK_RATE 4000000  // MCP40210 max clock rate for daisy-chain configuration is 5.8 MHz
#define WRITE_COMMAND 0x10
#define P0_WRITE_COMMAND WRITE_COMMAND | 0x01
#define P1_WRITE_COMMAND WRITE_COMMAND | 0x02

// Button-input macros

#define LAYER_BTN_MASK 0x01
#define NEURON_BTN_MASK 0x02
#define SYNAPSE_BTN_MASK 0x04
#define UP_BTN_MASK 0x08
#define DOWN_BTN_MASK 0x10
#define WEIGHT_RESET_BTN_MASK 0x20
#define MIN_BTN_MASK 0x40
#define MAX_BTN_MASK 0x80

// Display macros
#define ARROW_CHAR (char)0x7f

// Pin selection macros

#define CHIP_SELECT_PIN 9

#define LAYER_BTN_PIN 32
#define NEURON_BTN_PIN 31
#define SYNAPSE_BTN_PIN 30
#define UP_BTN_PIN 29
#define DOWN_BTN_PIN 28
#define WEIGHT_RESET_BTN_PIN 27
#define MIN_BTN_PIN 26
#define MAX_BTN_PIN 25

#define LCD_RS_PIN 38
#define LCD_EN_PIN 37
#define LCD_D4_PIN 36
#define LCD_D5_PIN 35
#define LCD_D6_PIN 34
#define LCD_D7_PIN 33

typedef struct NeuronLayer {
   int inputsPerNeuron;
   int neuronCount;
   byte **weights;
} NeuronLayer;

typedef struct NeuralNetwork {
   int layerCount;
   NeuronLayer **layers;
   byte *weights;
   int weightCount;
} NeuralNetwork;

void setDigitalPots(byte data[]);
void readButtonInput(int *buttonState);
void executeButtonInstruction(int *buttonState, int *layerIndex, int *neuronIndex, int *synapseIndex, byte weight);

const byte _P0WriteCommand = P0_WRITE_COMMAND;
const byte _P1WriteCommand = P1_WRITE_COMMAND;
const byte _default_weight = 0;

LiquidCrystal lcd(LCD_RS_PIN, LCD_EN_PIN, LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN);

void setup() {
  SPI.begin();
  pinMode(CHIP_SELECT_PIN, OUTPUT);
  digitalWrite(CHIP_SELECT_PIN, HIGH);

  pinMode(LAYER_BTN_PIN, INPUT_PULLUP);
  pinMode(NEURON_BTN_PIN, INPUT_PULLUP);
  pinMode(SYNAPSE_BTN_PIN, INPUT_PULLUP);
  pinMode(UP_BTN_PIN, INPUT_PULLUP);
  pinMode(DOWN_BTN_PIN, INPUT_PULLUP);
  pinMode(WEIGHT_RESET_BTN_PIN, INPUT_PULLUP);
  pinMode(MIN_BTN_PIN, INPUT_PULLUP);
  pinMode(MAX_BTN_PIN, INPUT_PULLUP);

  lcd.begin(16, 2);
}

void loop()
{
  char serialDebugBuff[100];
  int buttonState = 0;
  int layerIndex = 0;
  int neuronIndex = 0;
  int synapseIndex = -1;
  
  // Set up MLP layers.
  byte weights[20];
  
  byte *l0w = weights;
  byte *layer_0_weights_by_neuron[] = {l0w, l0w+4, l0w+8, l0w+12};
  NeuronLayer layer0 = { .inputsPerNeuron = 4, .neuronCount = 4, .weights = layer_0_weights_by_neuron };
  
  byte *l1w = l0w + (layer0.inputsPerNeuron * layer0.neuronCount);
  byte *layer_1_weights_by_neuron[] = {l1w }; //, l1w+4, l1w+8, l1w+12};
  NeuronLayer layer1 = { .inputsPerNeuron = 4, .neuronCount = 1, .weights = layer_1_weights_by_neuron };

//  byte *l2w = l1w + (layer1.inputsPerNeuron * layer1.neuronCount);
//  byte *layer_2_weights_by_neuron[] = {l2w, l2w+4, l2w+8};
//  NeuronLayer layer2 = { .inputsPerNeuron = 4, .neuronCount = 3, .weights = layer_2_weights_by_neuron };
  
  NeuronLayer *layers[] = { &layer0, &layer1 }; //, &layer2 };
  NeuralNetwork MLP = { .layerCount = 2, .layers = layers, .weights = weights, .weightCount = 20 };

  for (int i; i < MLP.weightCount; i++)
    weights[i] = _default_weight;
  
  // Main control loop.
  while (1)
  {
    readButtonInput(&buttonState);
    
    executeButtonInstruction(&buttonState, &layerIndex, &neuronIndex, &synapseIndex, &MLP);
    
    updateDisplay(layerIndex, neuronIndex, synapseIndex, &MLP);
    
    setDigitalPots(MLP.weights, MLP.weightCount);
  }

}

void updateDisplay(int layerIndex, int neuronIndex, int synapseIndex, NeuralNetwork *MLP)
{
  char buff[33];
  char line1[17];
  char line2[17];
  
  sprintf(line1, "Neuron %d_%d", layerIndex, neuronIndex);

  byte *weights = MLP->layers[layerIndex]->weights[neuronIndex];
  int synapseCount = MLP->layers[layerIndex]->inputsPerNeuron;
  int length = min(synapseCount, 4);
  for (int i = 0; i < length; i++)
  {
    // sprintf() will leave a '\0' at the end of each weight readout, but we will
    //   overwrite all of them except for the last one, so it's okay.
    char *startPosition = line2 + (4*i);
    sprintf(startPosition, "%3d ", weights[i]);

    // Add an arrow indicator if this weight is selected for adjustment.
    *(startPosition + 3) = synapseIndex < 0 || synapseIndex == i ? ARROW_CHAR : ' ';
  }
  
  lcd.setCursor(0,0);
  lcd.print(line1);
  
  lcd.setCursor(0,1);
  lcd.print(line2);
}

void readButtonInput(int *buttonState)
{
   if (digitalRead(LAYER_BTN_PIN) == LOW)
   {
      *buttonState |= LAYER_BTN_MASK;
      longDebounce();
   }
   else if (digitalRead(NEURON_BTN_PIN) == LOW)
   {
      *buttonState |= NEURON_BTN_MASK;
      longDebounce();
   }
   else if (digitalRead(SYNAPSE_BTN_PIN) == LOW)
   {
      *buttonState |= SYNAPSE_BTN_MASK;
      longDebounce();
   }
   else if (digitalRead(UP_BTN_PIN) == LOW)
   {
      *buttonState |= UP_BTN_MASK;
      shortDebounce();
   }
   else if (digitalRead(DOWN_BTN_PIN) == LOW)
   {
      *buttonState |= DOWN_BTN_MASK;
      shortDebounce();
   }
   else if (digitalRead(WEIGHT_RESET_BTN_PIN) == LOW)
   {
      *buttonState |= WEIGHT_RESET_BTN_MASK;
      longDebounce();
   }
   else if (digitalRead(MIN_BTN_PIN) == LOW)
   {
      *buttonState |= MIN_BTN_MASK;
      longDebounce();
   }
   else if (digitalRead(MAX_BTN_PIN) == LOW)
   {
      *buttonState |= MAX_BTN_MASK;
      longDebounce();
   }
}

int min(int a, int b)
{
  return (a < b) ? a : b;
}

void shortDebounce()
{
   delay(10);
}

void longDebounce()
{
   delay(200);
}

void executeButtonInstruction(int *buttonState, int *layerIndex, int *neuronIndex, int *synapseIndex, NeuralNetwork *MLP)
{
   if (*buttonState == 0)
   {
      return;
   }
   else if (*buttonState & LAYER_BTN_MASK)
   {
      if (*layerIndex < (MLP->layerCount - 1))
         *layerIndex += 1;
      else
         *layerIndex = 0;

      *neuronIndex = 0;
      *synapseIndex = -1;
   }
   else if (*buttonState & NEURON_BTN_MASK)
   {
      if (*neuronIndex < (MLP->layers[*layerIndex]->neuronCount - 1))
         *neuronIndex += 1;
      else
         *neuronIndex = 0;
      
      *synapseIndex = -1;
   }
   else if (*buttonState & SYNAPSE_BTN_MASK)
   {
      if (*synapseIndex < (MLP->layers[*layerIndex]->inputsPerNeuron - 1))
         *synapseIndex += 1;
      else 
         *synapseIndex = -1;
   }
   else if (*buttonState & UP_BTN_MASK)
   {
      byte *weights = MLP->layers[*layerIndex]->weights[*neuronIndex];
      adjustSynapses(*synapseIndex, MLP->layers[*layerIndex]->inputsPerNeuron, weights, &incrementWeight);
   }
   else if (*buttonState & DOWN_BTN_MASK)
   {      
      byte *weights = MLP->layers[*layerIndex]->weights[*neuronIndex];
      adjustSynapses(*synapseIndex, MLP->layers[*layerIndex]->inputsPerNeuron, weights, &decrementWeight);
   }
   else if (*buttonState & WEIGHT_RESET_BTN_MASK)
   {
      byte *weights = MLP->layers[*layerIndex]->weights[*neuronIndex];
      adjustSynapses(*synapseIndex, MLP->layers[*layerIndex]->inputsPerNeuron, weights, &resetWeight);
   }
   else if (*buttonState & MIN_BTN_MASK)
   {
      byte *weights = MLP->layers[*layerIndex]->weights[*neuronIndex];
      adjustSynapses(*synapseIndex, MLP->layers[*layerIndex]->inputsPerNeuron, weights, &setToMinWeight);
   }
   else if (*buttonState & MAX_BTN_MASK)
   {
      byte *weights = MLP->layers[*layerIndex]->weights[*neuronIndex];
      adjustSynapses(*synapseIndex, MLP->layers[*layerIndex]->inputsPerNeuron, weights, &setToMaxWeight);
   }

   *buttonState = 0;
}

void adjustSynapses(int synapseIndex, int inputsPerNeuron, byte* weights, void (*adjustmentFunction)(byte*))
{
  if (synapseIndex >= 0)
    (*adjustmentFunction)(weights + synapseIndex);
  else
    adjustAllWeights(weights, inputsPerNeuron, adjustmentFunction);
}

void adjustAllWeights(byte *weights, int length, void (*adjustmentFunction)(byte*))
{
    for (int i = 0; i < length; i++)
      (*adjustmentFunction)(weights+i);
}

void incrementWeight(byte *weight)
{
  if (*weight < 255)
     *weight += 1;  
}

void decrementWeight(byte *weight)
{
  if (*weight > 0)
     *weight -= 1;  
}

void resetWeight(byte *weight)
{
  *weight = 128;
}

void setToMinWeight(byte *weight)
{
  *weight = 0;
}

void setToMaxWeight(byte *weight)
{
  *weight = 255;
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

  // This might not be required, but give the pots some delay to set wiper values.
//  delay(50);

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
}
