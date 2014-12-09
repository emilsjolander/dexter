package main

import (
	"fmt"

	"github.com/emilsjolander/dexter/hmm"
)

func main() {
	model := hmm.New(
		hmm.NewState(0.5, []hmm.Distribution{{0.9, 0.1}}, hmm.Distribution{0.9, 0.1}),
		hmm.NewState(0.5, []hmm.Distribution{{0.1, 0.9}}, hmm.Distribution{0.1, 0.9}),
	)

	seq := []hmm.Point{{0}, {0}, {0}, {1}, {1}, {1}}

	for i := 0; i < 10; i++ {
		model.Train(seq)
	}

	fmt.Println(model.Probability([]hmm.Point{{0}, {0}, {0}, {1}, {1}, {1}}))
	fmt.Println(model.Probability([]hmm.Point{{1}, {1}, {1}, {0}, {0}, {0}}))
}
