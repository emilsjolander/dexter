package knn

import (
	"math"
	"reflect"
	"testing"
)

func TestEuclideanDistance(t *testing.T) {
	if EuclideanDistance([]float64{1, 1}, []float64{2, 2}) != math.Sqrt(2) {
		t.Fail()
	}
}

func TestClassify(t *testing.T) {
	knn := New()
	knn.Train([]float64{1, 1}, "one")
	knn.Train([]float64{2, 2}, "two")

	if class, err := knn.Classify([]float64{1.1, 1.1}, 1); class != "one" {
		t.Errorf("Failed to classify class one: class = %v, error = %v", class, err)
		return
	}

	if class, err := knn.Classify([]float64{1.8, 1.8}, 1); class != "two" {
		t.Errorf("Failed to classify class two: class = %v, error = %v", class, err)
		return
	}
}

func TestInsert(t *testing.T) {
	var node *kdtree = &kdtree{
		value: []float64{0, 0},
		class: "",
		depth: 0,
	}
	node.insert([]float64{1, 1}, "")
	node.insert([]float64{-1, -1}, "")

	if !reflect.DeepEqual(node, &kdtree{
		value: []float64{0, 0},
		class: "",
		depth: 0,
		left: &kdtree{
			value: []float64{-1, -1},
			class: "",
			depth: 1,
		},
		right: &kdtree{
			value: []float64{1, 1},
			class: "",
			depth: 1,
		},
	}) {
		t.Fail()
	}
}
