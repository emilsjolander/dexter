package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/emilsjolander/dexter/nba"
)

func main() {
	classifier := nba.NewMultinomial()

	training := readData("20news-bydate/20news-bydate-train")
	testing := readData("20news-bydate/20news-bydate-test")

	if err := classifier.Fit(training); err != nil {
		panic(err)
	}

	total := 0
	correct := 0
	for class, docs := range testing {
		for _, doc := range docs {
			total++
			result, err := classifier.Classify(doc)
			if err == nil && class == result {
				correct++
			}
		}
	}

	fmt.Println("Total: ", total)
	fmt.Println("Correct: ", correct)
	fmt.Println("%: ", 100*(float64(correct)/float64(total)))
}

func readData(dirName string) map[string][]string {
	children, err := ioutil.ReadDir(dirName)
	if err != nil {
		panic(err)
	}

	data := make(map[string][]string)

	for _, child := range children {
		if !strings.HasPrefix(child.Name(), ".") {
			docs := make([]string, 0)

			docFiles, err := ioutil.ReadDir(dirName + "/" + child.Name())
			if err != nil {
				panic(err)
			}
			for _, f := range docFiles {
				contentFile, err := os.Open(dirName + "/" + child.Name() + "/" + f.Name())
				if err != nil {
					panic(err)
				}
				content, err := ioutil.ReadAll(contentFile)
				if err != nil {
					panic(err)
				}
				docs = append(docs, string(content))
				contentFile.Close()
			}

			data[child.Name()] = docs
		}
	}

	return data
}
