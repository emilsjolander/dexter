package hmm

type (
	Probability  float64
	Distribution []Probability
	Point        []int
	Sequence     []Point

	State struct {
		// Probability to start in this state
		Pi Probability
		// Probability distribution over all states
		Transitions Distribution
		// Multi dimensiona probability distribution for emission
		Emissions []Distribution
		// Index of state in hmm
		Index int
	}

	Hmm []*State
)

func New(states ...*State) Hmm {
	for i, s := range states {
		s.Index = i
	}
	hmm := Hmm(states)
	hmm.normalize()
	return hmm
}

func NewState(pi Probability, emissions []Distribution, transitions Distribution) *State {
	return &State{
		Pi:          pi,
		Emissions:   emissions,
		Transitions: transitions,
	}
}

func (s *State) Probability(p Point) Probability {
	var prob Probability
	for i, v := range p {
		if i == 0 {
			prob = s.Emissions[i][v]
		} else {
			prob *= s.Emissions[i][v]
		}
	}
	return prob
}

func (hmm Hmm) Train(seq Sequence) {
	stateProbability := func(state *State, time int) Probability {
		return (hmm.Forward(seq[:time+1], state) * hmm.Backward(seq[time+1:], state)) / hmm.Probability(seq)
	}
	transitionProbability := func(from *State, to *State, time int) Probability {
		return hmm.Forward(seq[:time+1], from) * from.Transitions[to.Index] * to.Probability(seq[time+1]) * hmm.Backward(seq[time+2:], to)
	}

	// Create a new empty hmm with the same structure as the current one.
	// This is so that probabilities can be calculated on the last iteration's
	// data while updating the values of the model.
	newHmm := hmm.emptyCopy()

	// Calculate the new initial distribution
	for state := range hmm {
		newHmm[state].Pi = stateProbability(hmm[state], 0)
	}

	// Calculate the new transition distributions
	for from := range hmm {
		for to := range hmm {
			var transitionsFrom Probability
			var transitionsBetween Probability
			for i := 0; i < len(seq)-1; i++ {
				transitionsFrom += stateProbability(hmm[from], i)
				transitionsBetween += transitionProbability(hmm[from], hmm[to], i)
			}
			newHmm[from].Transitions[to] = transitionsBetween / transitionsFrom
		}
	}

	// Calculate the new emission distributions
	for state := range hmm {
		var totalStateProbability Probability
		for t := 0; t < len(seq); t++ {
			totalStateProbability += stateProbability(hmm[state], t)
		}

		for dimension := range hmm[state].Emissions {
			for emission := range hmm[state].Emissions[dimension] {
				var totalEmissionProbability Probability
				for t := 0; t < len(seq); t++ {
					totalEmissionProbability += stateProbability(hmm[state], t) * hmm[state].Emissions[dimension][emission]
				}
				newHmm[state].Emissions[dimension][emission] = totalEmissionProbability / totalStateProbability
			}
		}
	}

	newHmm.normalize()

	// Copy new values back into original hmm structure
	for state := range hmm {
		hmm[state] = newHmm[state]
	}
}

func (hmm Hmm) Probability(seq Sequence) Probability {
	var p Probability
	for _, s := range hmm {
		p += hmm.Forward(seq, s)
	}
	return p
}

func (hmm Hmm) Viterbi(seq Sequence, state *State) []*State {
	type path struct {
		p      Probability
		states []*State
	}

	// map[stateIndex][timeStep] = probability
	memory := make(map[int]map[int]path)
	var f func(Sequence, *State) path

	f = func(seq Sequence, state *State) path {
		t := len(seq) - 1
		var maxP Probability
		var bestPath path

		if _, exists := memory[state.Index]; !exists {
			memory[state.Index] = make(map[int]path)
		}

		// DP step
		if p, ok := memory[state.Index][t]; ok {
			return p
		}

		// Bottom of the recursion
		if t == 0 {
			return path{
				p:      state.Pi * state.Probability(seq[0]),
				states: []*State{state},
			}
		}

		// Perform recursive step
		for i := 0; i < len(hmm); i++ {
			p := f(seq[:t], hmm[i])
			if p.p > maxP {
				maxP = p.p
				bestPath.p = p.p * hmm[i].Transitions[state.Index] * state.Probability(seq[t])
				bestPath.states = append(p.states, state)
			}
		}

		memory[state.Index][t] = bestPath
		return bestPath
	}

	return f(seq, state).states
}

func (hmm Hmm) Forward(seq Sequence, state *State) Probability {
	// map[stateIndex][timeStep] = probability
	memory := make(map[int]map[int]Probability)
	var f func(Sequence, *State) Probability

	f = func(seq Sequence, state *State) Probability {
		t := len(seq) - 1
		var p Probability

		if _, exists := memory[state.Index]; !exists {
			memory[state.Index] = make(map[int]Probability)
		}

		// DP step
		if p, ok := memory[state.Index][t]; ok {
			return p
		}

		// Bottom of the recursion
		if t == 0 {
			return state.Pi * state.Probability(seq[0])
		}

		// Perform recursive step
		for i := 0; i < len(hmm); i++ {
			p += f(seq[:t], hmm[i]) * hmm[i].Transitions[state.Index] * state.Probability(seq[t])
		}

		memory[state.Index][t] = p
		return p
	}

	return f(seq, state)
}

func (hmm Hmm) Backward(seq Sequence, state *State) Probability {
	// map[stateIndex][timeStep] = probability
	memory := make(map[int]map[int]Probability)
	var f func(Sequence, *State) Probability

	f = func(seq Sequence, state *State) Probability {
		t := len(seq) - 1
		var p Probability

		if _, exists := memory[state.Index]; !exists {
			memory[state.Index] = make(map[int]Probability)
		}

		// DP step
		if p, ok := memory[state.Index][t]; ok {
			return p
		}

		// Bottom of the recursion
		if len(seq) == 0 {
			return 1
		}

		for i := 0; i < len(hmm); i++ {
			p += state.Transitions[i] * hmm[i].Probability(seq[0]) * f(seq[1:], hmm[i])
		}

		memory[state.Index][t] = p
		return p
	}

	return f(seq, state)
}

func (hmm Hmm) emptyCopy() Hmm {
	newHmm := Hmm(make([]*State, len(hmm)))
	for i := range newHmm {
		newHmm[i] = &State{
			Pi:          0,
			Transitions: make([]Probability, len(hmm)),
			Emissions:   make([]Distribution, len(hmm[i].Emissions)),
			Index:       i,
		}
		for j := range newHmm[i].Emissions {
			newHmm[i].Emissions[j] = make([]Probability, len(hmm[i].Emissions[j]))
		}
	}
	return newHmm
}

// Make sure all probabilities sum to 1
func (hmm Hmm) normalize() {
	var piSum Probability
	for _, state := range hmm {
		piSum += state.Pi
	}
	for _, state := range hmm {
		state.Pi /= piSum
		state.Transitions.normalize()

		for dimension := range state.Emissions {
			state.Emissions[dimension].normalize()
		}
	}
}

func (dist Distribution) normalize() {
	var sum Probability
	for _, p := range dist {
		sum += p
	}
	for i, p := range dist {
		dist[i] = p / sum
	}
}
