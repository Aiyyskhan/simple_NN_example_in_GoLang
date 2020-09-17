/*
 * Double layer perceptron example
 * Sigmoid activation function
 * Backpropagation
 */

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/Aiyyskhan/simple_NN_example_in_GoLang/dataPackage"
)

const (
	// гиперпараметры
	numIn            = 7
	numHidd          = 8
	numOut           = 4
	learningRate     = 0.7 //скорость обучения
	initialWeightMax = 0.5 //максимум диапазона случайных значений стартовых весов
	momentum         = 0.9 // инерция
	success          = 0.005
	patternCount     = 10
)

var out [numOut]float64 //выход сети
var hNeuron [numHidd][2][numIn + 1]float64
var oNeuron [numOut][2][numHidd + 1]float64
var changeHiddNeuron [numHidd][numIn + 1]float64
var changeOutNeuron [numOut][numHidd + 1]float64

var (
	i int
	j int
	p int
	q int
	r int
)

var randomizedIndex [patternCount]int
var epoch uint64

func main() {
	setup()

	fmt.Println("First test: ")
	testing()

	trainNN() //обучение

	fmt.Println("")
	fmt.Println("Final test: ")
	testing()
}

func setup() {
	for i := 0; i < numHidd; i++ {
		for j := 0; j <= numIn; j++ {
			changeHiddNeuron[i][j] = 0.0
			hNeuron[i][0][j] = 2.0 * ((float64(rand.Intn(100)) / 100) - 0.5) * initialWeightMax
		}
	}
	for i := 0; i < numOut; i++ {
		for j := 0; j <= numHidd; j++ {
			changeOutNeuron[i][j] = 0.0
			oNeuron[i][0][j] = 2.0 * ((float64(rand.Intn(100)) / 100) - 0.5) * initialWeightMax
		}
	}

	for p := 0; p < patternCount; p++ {
		randomizedIndex[p] = p
	}
}

func neuron() { //логика нейрона
	for i := 0; i < numOut; i++ {
		out[i] = 0.0
	}

	var s [numHidd]float64
	for i := 0; i < numHidd; i++ {
		s[i] = hNeuron[i][0][numIn] //bias
		for j := 0; j < numIn; j++ {
			s[i] += hNeuron[i][0][j] * hNeuron[i][1][j]
		}
	}
	for i := 0; i < numOut; i++ {
		out[i] = oNeuron[i][0][numHidd] //bias
		for j := 0; j < numHidd; j++ {
			oNeuron[i][1][j] = activateFunc(s[j])
			out[i] += oNeuron[i][0][j] * oNeuron[i][1][j]
		}
		out[i] = activateFunc(out[i])
	}
}

func activateFunc(arg float64) float64 {
	result := 1.0 / (1.0 + math.Exp(-arg))
	return result
}

func trainNN() {

	fmt.Println("Start train!")
	it := 0 //итерации
	var err float64
	var hiddDelta [numHidd]float64
	var outDelta [numOut]float64
	for epoch = 1; epoch < 1000000; epoch++ {
		it++ //увеличиваем на 1 итерацию
		for p := 0; p < patternCount; p++ {
			q = rand.Intn(patternCount)
			r = randomizedIndex[p]
			randomizedIndex[p] = randomizedIndex[q]
			randomizedIndex[q] = r
		}
		err = 0.0

		for q := 0; q < patternCount; q++ {
			p = randomizedIndex[q]

			//вводим обучающие значения на вход
			for i := 0; i < numHidd; i++ {
				for j := 0; j < numIn; j++ {
					hNeuron[i][1][j] = dataPackage.Input[p][j]
				}
			}

			neuron() //обрабатываем на нейроне

			//вычисляем ошибку на выходе
			for i := 0; i < numOut; i++ {
				outDelta[i] = (dataPackage.Target[p][i] - out[i]) * out[i] * (1.0 - out[i])
				err += 0.5 * (dataPackage.Target[p][i] - out[i]) * (dataPackage.Target[p][i] - out[i])
			}

			//обратное распространение ошибки на скрытый слой
			for i := 0; i < numHidd; i++ {
				accum := 0.0
				for j := 0; j < numOut; j++ {
					accum += oNeuron[j][0][i] * outDelta[j]
				}
				hiddDelta[i] = accum * oNeuron[j][1][i] * (1.0 - oNeuron[j][1][i])
			}

			//коррекция весов скрытого слоя
			for i := 0; i < numHidd; i++ {
				changeHiddNeuron[i][numIn] = learningRate*hiddDelta[i] + momentum*changeHiddNeuron[i][numIn]
				hNeuron[i][0][numIn] += changeHiddNeuron[i][numIn]
				for j := 0; j < numIn; j++ {
					changeHiddNeuron[i][j] = learningRate*hNeuron[i][1][j]*hiddDelta[i] + momentum*changeHiddNeuron[i][j]
					hNeuron[i][0][j] += changeHiddNeuron[i][j]
				}
			}

			//коррекция весов выходного слоя
			for i := 0; i < numOut; i++ {
				changeOutNeuron[i][numHidd] = learningRate*outDelta[i] + momentum*changeOutNeuron[i][numHidd]
				oNeuron[i][0][numHidd] += changeOutNeuron[i][numHidd]
				for j := 0; j < numHidd; j++ {
					changeOutNeuron[i][j] = learningRate*oNeuron[i][1][j]*outDelta[i] + momentum*changeOutNeuron[i][j]
					oNeuron[i][0][j] += changeOutNeuron[i][j]
				}
			}
		}

		if it == 100 {
			fmt.Println("Epoch:", epoch)
			fmt.Println("Error:", err)
			fmt.Println()
			it = 0
		}

		if err < success {
			break
		} //пока error не равно Success, выполняем обучение
	}
	fmt.Println("End train!")
	fmt.Println("Epoch:", epoch)
	fmt.Println("Error:", err)
}

func testing() {
	for p := 0; p < patternCount; p++ {
		fmt.Println()
		fmt.Println("Pattern:", p)
		fmt.Println("Input")
		for i := 0; i < numIn; i++ {
			fmt.Println(dataPackage.Input[p][i])
			fmt.Println(" ")
		}
		fmt.Println("  Target ")
		for i := 0; i < numOut; i++ {
			fmt.Println(dataPackage.Target[p][i])
			fmt.Println(" ")
		}
		for i := 0; i < numHidd; i++ {
			for j := 0; j < numIn; j++ {
				hNeuron[i][1][j] = dataPackage.Input[p][j]
			}
		}
		neuron()
		fmt.Println("Output")
		for i := 0; i < numOut; i++ {
			fmt.Println(out[i])
		}
	}
	fmt.Println()
}
