package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/emilsjolander/dexter/ann"
)

func main() {
	net := ann.NewFeedForward(0.1,
		ann.NewInputLayer(64),
		ann.NewLayer(64, new(ann.SigmoidActivation)),
		ann.NewLayer(10, new(ann.SigmoidActivation)),
	)

	trainingInputs, trainingClasses := readData("optdigits.tra")
	testingInputs, testingClasses := readData("optdigits.tes")

	for j := 0; j < 100; j++ {
		for i := range trainingInputs {
			_, err := net.Train(trainingInputs[i], makeOutput(trainingClasses[i]))
			if err != nil {
				panic(err)
			}
		}
	}

	correct := 0
	for i := range testingInputs {
		out, err := net.Predict(testingInputs[i])
		if err != nil {
			panic(err)
		}
		if checkEquals(out, testingClasses[i]) {
			correct++
		}
	}

	fmt.Println("Total: ", len(testingInputs))
	fmt.Println("Correct: ", correct)
	fmt.Println("%: ", 100*(float64(correct)/float64(len(testingInputs))))
}

func makeOutput(o float64) []float64 {
	out := make([]float64, 10)
	out[int(o)] = 1
	return out
}

func checkEquals(output []float64, class float64) bool {
	largestIndex := 0
	for i, o := range output {
		if o > output[largestIndex] {
			largestIndex = i
		}
	}
	return int(class) == largestIndex
}

func readData(fileName string) ([][]float64, []float64) {
	file, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}
	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		panic(err)
	}

	var classes []float64
	var inputs [][]float64
	for _, row := range records {
		var input []float64
		for _, item := range row[:len(row)-1] {
			i, err := strconv.ParseFloat(item, 64)
			if err != nil {
				panic(err)
			}
			input = append(input, i/100)
		}
		inputs = append(inputs, input)

		class, err := strconv.ParseFloat(row[len(row)-1], 64)
		if err != nil {
			panic(err)
		}
		classes = append(classes, class)
	}

	return inputs, classes
}
