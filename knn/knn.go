package knn

import (
	"errors"
	"math"
)

type Knn struct {
	Dimensions int
	Distance   func([]float64, []float64) float64
	root       *kdtree
}

type kdtree struct {
	value []float64
	class string
	depth int
	left  *kdtree
	right *kdtree
}

var WrongDimensionError = errors.New("Dimensionality of data does not match that specified in construction of knn")
var NotTrainedError = errors.New("Knn has not been trained with any data")

func New() *Knn {
	return new(Knn)
}

func EuclideanDistance(p1 []float64, p2 []float64) float64 {
	var distance float64
	for i := 0; i < len(p1); i++ {
		distance += math.Pow(p1[i]-p2[i], 2)
	}
	return math.Sqrt(distance)
}

func (knn *Knn) Train(point []float64, class string) error {
	if knn.Dimensions == 0 {
		knn.Dimensions = len(point)
	} else if len(point) != knn.Dimensions {
		return WrongDimensionError
	}

	if knn.root == nil {
		knn.root = &kdtree{
			value: point,
			class: class,
			depth: 0,
		}
	} else {
		knn.root.insert(point, class)
	}

	return nil
}

func (knn *Knn) Classify(point []float64, k int) (string, error) {
	if knn.root == nil {
		return "", NotTrainedError
	}
	if len(point) != knn.Dimensions {
		return "", WrongDimensionError
	}

	// Set default distnace function if none is present
	if knn.Distance == nil {
		knn.Distance = EuclideanDistance
	}

	nearest := make([]*kdtree, k)
	knn.nearestNieghbours(knn.root, point, nearest)

	// Gather the votes
	votes := make(map[string]int)
	for _, n := range nearest {
		votes[n.class]++
	}

	// Find the class with the largest vote
	vote := 0
	class := ""
	for k, v := range votes {
		if vote < v {
			vote = v
			class = k
		}
	}

	return class, nil
}

func (knn *Knn) nearestNieghbours(node *kdtree, point []float64, nearest []*kdtree) {
	if node == nil {
		return
	}
	axis := node.depth % len(point)

	// Navigate to the bottom of the tree
	if point[axis] < node.value[axis] {
		knn.nearestNieghbours(node.left, point, nearest)
	} else {
		knn.nearestNieghbours(node.right, point, nearest)
	}

	// While recursing up check if this node is closer than any other node in the list
	max, i := knn.maxDist(point, nearest)
	dist := knn.Distance(point, node.value)
	if dist < max {
		nearest[i] = node
	}

	// Update max
	max, _ = knn.maxDist(point, nearest)

	// Check if the hypersphere around point crosses this hyperplane, in that case traverse the other branch
	if max > math.Abs(point[axis]-node.value[axis]) {
		if point[axis] < node.value[axis] {
			knn.nearestNieghbours(node.right, point, nearest)
		} else if point[axis] > node.value[axis] {
			knn.nearestNieghbours(node.left, point, nearest)
		}
	}
}

func (knn *Knn) maxDist(point []float64, nearest []*kdtree) (float64, int) {
	var max float64 = -1
	var maxIndex int
	for i, n := range nearest {
		if n == nil {
			return math.MaxFloat64, i
		}
		dist := knn.Distance(point, n.value)
		if dist > max {
			max = dist
			maxIndex = i
		}
	}
	return max, maxIndex
}

func (node *kdtree) insert(point []float64, class string) {
	axis := node.depth % len(point)

	if point[axis] < node.value[axis] {
		if node.left == nil {
			node.left = &kdtree{
				value: point,
				class: class,
				depth: node.depth + 1,
			}
		} else {
			node.left.insert(point, class)
		}
	} else {
		if node.right == nil {
			node.right = &kdtree{
				value: point,
				class: class,
				depth: node.depth + 1,
			}
		} else {
			node.right.insert(point, class)
		}
	}
}
