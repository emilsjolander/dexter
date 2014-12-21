package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"strconv"

	"github.com/emilsjolander/dexter/knn"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	trainingInputs, trainingClasses := readData("optdigits.tra")
	testingInputs, testingClasses := readData("optdigits.tes")

	knn := knn.New(64, knn.EuclideanDistance)
	err := knn.Fit(trainingInputs, trainingClasses)
	if err != nil {
		panic(err)
	}

	correct := 0
	for i := range testingInputs {
		class, err := knn.Classify(testingInputs[i], 1)
		if err != nil {
			panic(err)
		}
		if class == testingClasses[i] {
			correct++
		}
	}

	fmt.Println("Total: ", len(testingInputs))
	fmt.Println("Correct: ", correct)
	fmt.Println("%: ", 100*(float64(correct)/float64(len(testingInputs))))
}

func readData(fileName string) ([]knn.Point, []string) {
	file, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}
	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		panic(err)
	}

	var classes []string
	var inputs []knn.Point
	for _, row := range records {
		var input knn.Point
		for _, item := range row[:len(row)-1] {
			i, err := strconv.ParseFloat(item, 64)
			if err != nil {
				panic(err)
			}
			input = append(input, i)
		}
		inputs = append(inputs, input)

		class, err := strconv.ParseInt(row[len(row)-1], 10, 64)
		if err != nil {
			panic(err)
		}
		classes = append(classes, fmt.Sprintf("%d", class))
	}

	return inputs, classes
}
