/*
 * Double layer perceptron example
 * Sigmoid activation function
 * Backpropagation
 */

package main

import (
  "D:/Golang_projects/simple_NN_example_in_GoLang/dataPackage/dataset"
  "fmt"
  "math/rand"
)

const ( 
  // гиперпараметры
  numIn = 4
  numHidd = 8
  numOut = 7
  learningRate = 0.7 //скорость обучения
  initialWeightMax = 0.5 //максимум диапазона случайных значений стартовых весов
  momentum = 0.9 // инерция
  success = 0.005
  patternCount = 10
)

var out [numOut]float64 //выход сети
var hNeuron [numHidd][2][numIn+1]float64
var oNeuron [numOut][2][numHidd+1]float64
var changeHiddNeuron [numHidd][numIn+1]float64
var changeOutNeuron [numOut][numHidd+1]float64

var (
  i int
  j int
  p int
  q int
  r int
)

var randomizedIndex [PatternCount]int
var epoch uint64
 
func main() {
}
 
func setup() {
  for i:=0; i<hiddQ; i++ {
    for j:=0; j<=inQ; j++ {
      changeHiddNeuron[i][j] = 0.0
      hNeuron[i][0][j] = 2.0 * ((float64(rand.Intn(100))/100) - 0.5) * InitialWeightMax
    }
  }
  for i:=0; i<outQ; i++ {
    for j:=0; j<=hiddQ; j++ {
      changeOutNeuron[i][j] = 0.0
      oNeuron[i][0][j] = 2.0 * ((float64(rand.Intn(100))/100) - 0.5) * InitialWeightMax
    }
  }
  
  for p:=0; p < PatternCount; p++ {    
    randomizedIndex[p] = p
  }
  
  Serial.println("First test: ");
  testing();
  
  trainNN(); //обучение
  
  Serial.println("");  
  Serial.println("Final test: ");
  testing();
}

func neuron(){ //логика нейрона
  for i:=0; i<outQ; i++ {
	  out[i] = 0.0
  }
    
  var s [hiddQ]float64
  for i:=0; i<hiddQ; i++ {
    s[i] = hNeuron[i][0][inQ] //bias
    for j:=0; j<inQ; j++ {
      s[i] += hNeuron[i][0][j] * hNeuron[i][1][j]
    }
  }
  for i:=0; i<outQ; i++ {
    out[i] = oNeuron[i][0][hiddQ] //bias
    for j:=0; j<hiddQ; j++ {
      oNeuron[i][1][j] = activateFunc(s[j])
      out[i] += oNeuron[i][0][j] * oNeuron[i][1][j]
    }
    out[i] = activateFunc(out[i])
  }
}

func activateFunc(arg float64)(float64){
  result := 1.0/(1.0 + exp(-arg))
  return result
}

func trainNN(){
  
  fmt.println("Start train!")
  it := 0 //итерации
  var error float64
  var hiddDelta [hiddQ]float64;
  var outDelta [outQ]float64
  for epoch=1; epoch < 1000000; epoch++ {
    it++ //увеличиваем на 1 итерацию
    for p = 0; p < PatternCount; p++ {
      q = rand.Intn(PatternCount)
      r = randomizedIndex[p]
      randomizedIndex[p] = randomizedIndex[q]
      randomizedIndex[q] = r
    }
    error = 0.0
    
    for q = 0; q < PatternCount; q++ {
      p = randomizedIndex[q]

      //вводим обучающие значения на вход
      for i=0; i<hiddQ; i++ {
        for j=0; j<inQ; j++ {
          hNeuron[i][1][j] = Input[p][j]
        }
      }
      
      neuron(); //обрабатываем на нейроне

      //вычисляем ошибку на выходе
      for(int i=0; i<outQ; i++){ 
        outDelta[i] = (Target[p][i] - out[i]) * out[i] * (1.0 - out[i]);
        error += 0.5 * (Target[p][i] - out[i]) * (Target[p][i] - out[i]); 
      }

      //обратное распространение ошибки на скрытый слой
      for(int i=0; i<hiddQ; i++){ 
        float accum = 0.0;
        for(int j=0; j<outQ; j++){
          accum += oNeuron[j][0][i] * outDelta[j];
        }
        hiddDelta[i] = accum * oNeuron[j][1][i] * (1.0 - oNeuron[j][1][i]);
      }

      //коррекция весов скрытого слоя
      for(int i=0; i<hiddQ; i++){ 
        changeHiddNeuron[i][inQ] = LearningRate * hiddDelta[i] + Momentum * changeHiddNeuron[i][inQ];
        hNeuron[i][0][inQ] += changeHiddNeuron[i][inQ];
        for(int j=0; j<inQ; j++){
          changeHiddNeuron[i][j] = LearningRate * hNeuron[i][1][j] * hiddDelta[i] + Momentum * changeHiddNeuron[i][j];
          hNeuron[i][0][j] += changeHiddNeuron[i][j];
        }
      }

      //коррекция весов выходного слоя
      for(int i=0; i<outQ; i++){ 
        changeOutNeuron[i][hiddQ] = LearningRate * outDelta[i] + Momentum * changeOutNeuron[i][hiddQ];
        oNeuron[i][0][hiddQ] += changeOutNeuron[i][hiddQ];
        for(int j=0; j<hiddQ; j++){
          changeOutNeuron[i][j] = LearningRate * oNeuron[i][1][j] * outDelta[i] + Momentum * changeOutNeuron[i][j];
          oNeuron[i][0][j] += changeOutNeuron[i][j];
        }
      }
    }

    if(it == 100){
      Serial.print("Epoch: ");
      Serial.println(epoch);
      Serial.print("Error: ");
      Serial.println(error, 5);
      Serial.println();
      it = 0;
    }
        
    if( error < Success ) break; //пока error не равно Success, выполняем обучение
  } 
  Serial.println("End train!");
  Serial.print("Epoch: ");
  Serial.println(epoch);
  Serial.print("Error: ");
  Serial.println(error, 5);
}

void testing(){
  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Pattern: ");
    Serial.println (p);      
    Serial.print ("  Input ");
    for( i = 0 ; i < inQ ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for( i = 0 ; i < outQ ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }
    for(int i=0; i<hiddQ; i++){
      for(int j=0; j<inQ; j++){
        hNeuron[i][1][j] = Input[p][j];
      }
    }
    neuron();
    Serial.print ("  Output ");
    for( i = 0 ; i < outQ ; i++ ) {       
      Serial.print (out[i], 5);
      Serial.print (" ");
    }
  }
  Serial.println();  
}

