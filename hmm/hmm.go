package dexter

type Hmm struct {
	Pi     []float64
	States []*State
	A      [][]float64
}

type State struct {
	B [][]float64
}

func (s *State) p(e []int) float64 {
	var prod float64
	for i := 0; i < len(e); i++ {
		prod *= s.B[i][e[i]]
	}
	return prod
}

func (hmm *Hmm) Train(seq [][]int) {
	// baum-welch
}

func (hmm *Hmm) Probability(seq [][]int) float64 {
	var p float64
	for _, s := range hmm.States {
		p += hmm.Forward(seq, s)
	}
	return p
}

func (hmm *Hmm) Viterbi(seq [][]int, state *State) []*State {
	type path struct {
		p      float64
		states []*State
	}

	// map[stateIndex][timeStep] = probability
	var memory map[int]map[int]path
	var f func([][]int, *State) path

	f = func(seq [][]int, state *State) path {
		stateIndex := hmm.indexOf(state)
		t := len(seq) - 1
		var maxP float64
		var bestPath path

		// DP step
		if p, ok := memory[stateIndex][t]; ok {
			return p
		}

		// Bottom of the recursion
		if t == 0 {
			return path{
				p:      hmm.Pi[stateIndex] * state.p(seq[0]),
				states: []*State{state},
			}
		}

		// Perform recursive step
		for i := 0; i < len(hmm.States); i++ {
			p := f(seq[:t], hmm.States[i])
			if p.p > maxP {
				maxP = p.p
				bestPath.p = p.p * hmm.A[i][stateIndex] * state.p(seq[t])
				bestPath.states = append(p.states, state)
			}
		}

		memory[stateIndex][t] = bestPath
		return bestPath
	}

	return f(seq, state).states
}

func (hmm *Hmm) Forward(seq [][]int, state *State) float64 {
	// map[stateIndex][timeStep] = probability
	var memory map[int]map[int]float64
	var f func([][]int, *State) float64

	f = func(seq [][]int, state *State) float64 {
		stateIndex := hmm.indexOf(state)
		t := len(seq) - 1
		var p float64

		// DP step
		if p, ok := memory[stateIndex][t]; ok {
			return p
		}

		// Bottom of the recursion
		if t == 0 {
			return hmm.Pi[stateIndex] * state.p(seq[0])
		}

		// Perform recursive step
		for i := 0; i < len(hmm.States); i++ {
			p += f(seq[:t], hmm.States[i]) * hmm.A[i][stateIndex] * state.p(seq[t])
		}

		memory[stateIndex][t] = p
		return p
	}

	return f(seq, state)
}

func (hmm *Hmm) Backward(seq [][]int, state *State) float64 {
	// map[stateIndex][timeStep] = probability
	var memory map[int]map[int]float64
	var f func([][]int, *State) float64

	f = func(seq [][]int, state *State) float64 {
		stateIndex := hmm.indexOf(state)
		t := len(seq) - 1
		var p float64

		// DP step
		if p, ok := memory[stateIndex][t]; ok {
			return p
		}

		// Bottom of the recursion
		if len(seq) == 0 {
			return 1
		}

		for i := 0; i < len(hmm.States); i++ {
			p += hmm.A[stateIndex][i] * hmm.States[i].p(seq[0]) * f(seq[1:], hmm.States[i])
		}

		memory[stateIndex][t] = p
		return p
	}

	return f(seq, state)
}

func (hmm *Hmm) indexOf(state *State) int {
	for i, s := range hmm.States {
		if s == state {
			return i
		}
	}
	panic("State does not exist")
}
