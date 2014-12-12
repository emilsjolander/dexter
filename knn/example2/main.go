package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/emilsjolander/dexter/knn"
)

func main() {
	knn := knn.New()

	trainingInputs, trainingClasses := readData("optdigits.tra")
	testingInputs, testingClasses := readData("optdigits.tes")

	for i := range trainingInputs {
		err := knn.Train(trainingInputs[i], fmt.Sprintf("%d", trainingClasses[i]))
		if err != nil {
			panic(err)
		}
	}

	correct := 0
	for i := range testingInputs {
		class, err := knn.Classify(testingInputs[i], 1)
		if err != nil {
			panic(err)
		}
		if class == fmt.Sprintf("%d", testingClasses[i]) {
			correct++
		}
	}

	fmt.Println("Total: ", len(testingInputs))
	fmt.Println("Correct: ", correct)
	fmt.Println("%: ", 100*(float64(correct)/float64(len(testingInputs))))
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
