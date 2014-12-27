package nba

import (
	"math"
	"strings"
)

type (
	Multinomial struct {
		// The number of words in the vocabulary
		vocabularySize int

		// The total number of words in a class
		classSize map[string]int

		// The prior probability of a class
		classPriors map[string]float64

		// The number of times a word has been seen in a class
		wordCount map[string]map[string]int
	}
)

func words(doc string) []string {
	words := strings.Split(string(doc), " ")
	for i, word := range words {
		words[i] = strings.ToLower(word)
	}
	return words
}

func NewMultinomial() *Multinomial {
	return &Multinomial{
		vocabularySize: 0,
		classSize:      make(map[string]int),
		classPriors:    make(map[string]float64),
		wordCount:      make(map[string]map[string]int),
	}
}

func (nba *Multinomial) Fit(data map[string][]string) error {
	if len(data) < 1 {
		return NoDataError
	}

	// Need to keep track of these to calculate class priors
	var totalDocs int = 0
	docsForClass := make(map[string]int)

	// Keep track of words that we have seen
	vocabulary := make(map[string]bool)

	for class, docs := range data {
		totalDocs += len(docs)
		docsForClass[class] = len(docs)

		for _, doc := range docs {
			nba.classSize[class] += len(doc)

			for _, word := range words(doc) {
				vocabulary[word] = true

				if _, ok := nba.wordCount[class]; !ok {
					nba.wordCount[class] = make(map[string]int)
				}
				nba.wordCount[class][word] += 1
			}
		}
	}

	// Calculate class priors
	for class, _ := range data {
		nba.classPriors[class] = float64(docsForClass[class]) / float64(totalDocs)
	}

	nba.vocabularySize = len(vocabulary)

	return nil
}

func (nba *Multinomial) Classify(doc string) (string, error) {
	var bestClass string
	var bestClassLogProbability float64 = -math.MaxFloat64

	for class, classPrior := range nba.classPriors {
		// Get the total class model for the point conditiond on this class
		var logSum float64 = 0
		for _, word := range words(doc) {
			logSum += math.Log(float64(nba.wordCount[class][word]+1) / float64(nba.classSize[class]+nba.vocabularySize))
		}

		// Bayes theorem: P(c|e) = (P(e|c)P(c)) / P(e)
		// We drop P(e) as it is constant
		logProbability := logSum + math.Log(classPrior)

		// Update current best class
		if logProbability > bestClassLogProbability {
			bestClassLogProbability = logProbability
			bestClass = class
		}
	}

	if bestClassLogProbability == -math.MaxFloat64 {
		return "", NoClassificationError
	}

	return bestClass, nil
}
