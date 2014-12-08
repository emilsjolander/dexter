package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/emilsjolander/dexter/knn"
)

func main() {
	dataFile, err := os.Open("iris.data")
	if err != nil {
		panic(err)
	}

	records, err := csv.NewReader(dataFile).ReadAll()
	if err != nil {
		panic(err)
	}

	Shuffle(records)
	classes := make([]string, len(records))
	data := make([][]float64, len(records))

	for i, row := range records {
		classes[i] = row[len(row)-1]
		data[i] = make([]float64, len(row)-1)
		for j, s := range row[:len(row)-1] {
			data[i][j], err = strconv.ParseFloat(s, 64)
			if err != nil {
				panic(err)
			}
		}
	}

	knn := knn.New()
	split := (len(data) / 4) * 3
	for i := 0; i < split; i++ {
		knn.Train(data[i], classes[i])
	}

	correct := 0
	for i := split; i < len(data); i++ {
		class, err := knn.Classify(data[i], 5)
		if err != nil {
			panic(err)
		}
		if class == classes[i] {
			correct++
		}
	}

	fmt.Println("Classified this many correctly: ", float64(correct)/float64(len(data)-split))
}

func Shuffle(a [][]string) {
	for i := range a {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}
