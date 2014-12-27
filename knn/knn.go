package knn

import (
	"errors"
	"math"
	"sort"
)

type (
	Point        []float64
	DistanceFunc func(Point, Point) float64

	value struct {
		point Point
		class string
	}

	Knn struct {
		Dimensionality int
		Distance       DistanceFunc
		root           *kdtree
	}

	kdtree struct {
		v     value
		depth int
		left  *kdtree
		right *kdtree
	}

	distClassPair struct {
		dist  float64
		class string
	}
)

var (
	WrongDimensionError = errors.New("Dimensionality of data does not match that specified in construction of knn")
	NotTrainedError     = errors.New("Knn has not been trained with any data")
	NoDataError         = errors.New("No data was suplied")
	LenMismatchError    = errors.New("Number of points does not match number of classes")
	NoDistanceFunction  = errors.New("A distance function must be specified")
)

// Construct a KNN with a certain dimensionality and distance functions
func New(dimensionality int, distance DistanceFunc) *Knn {
	return &Knn{
		Dimensionality: dimensionality,
		Distance:       distance,
	}
}

// Standard k-dimensional euclidean distance function
func EuclideanDistance(p1 Point, p2 Point) float64 {
	var distance float64
	for i := 0; i < len(p1); i++ {
		distance += (p1[i] - p2[i]) * (p1[i] - p2[i])
	}
	return math.Sqrt(distance)
}

// Standard k-dimensional manhattan distance function
func ManhattanDistance(p1 Point, p2 Point) float64 {
	var distance float64
	for i := 0; i < len(p1); i++ {
		distance += math.Abs(p1[i] - p2[i])
	}
	return distance
}

// Train the KNN with a data set consisting of a map from classes to set of points for that class
func (knn *Knn) Fit(points []Point, classes []string) error {
	if len(points) == 0 {
		return NoDataError
	}
	if len(points) != len(classes) {
		return LenMismatchError
	}

	// Gather values
	values := make([]value, 0)
	for i := range points {
		if len(points[i]) != knn.Dimensionality {
			return WrongDimensionError
		}
		values = append(values, value{
			point: points[i],
			class: classes[i],
		})
	}

	knn.root = &kdtree{depth: 0}
	insert(knn.root, values)

	return nil
}

// Classify a point using the trained KNN
func (knn *Knn) Classify(point Point, k int) (string, error) {
	if knn.root == nil {
		return "", NotTrainedError
	} else if len(point) != knn.Dimensionality {
		return "", WrongDimensionError
	} else if knn.Distance == nil {
		return "", NoDistanceFunction
	}

	nearest := make([]*distClassPair, k)
	knn.nearestNieghbours(knn.root, point, nearest)

	// Gather the votes
	votes := make(map[string]int)
	for _, n := range nearest {
		if n != nil {
			votes[n.class]++
		}
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

func (knn *Knn) nearestNieghbours(node *kdtree, point Point, nearest []*distClassPair) {
	if node == nil {
		return
	}
	axis := node.depth % len(point)
	var otherBranch *kdtree

	// Navigate to the bottom of the tree
	if point[axis] < node.v.point[axis] {
		knn.nearestNieghbours(node.left, point, nearest)
		otherBranch = node.right
	} else {
		knn.nearestNieghbours(node.right, point, nearest)
		otherBranch = node.left
	}

	// While recursing up check if this node is closer than any other node in the list
	dist := knn.Distance(point, node.v.point)
	max, i := maxDist(nearest)
	if dist < max {
		nearest[i] = &distClassPair{dist, node.v.class}
		max, _ = maxDist(nearest)
	}

	// Check if the hypersphere around point crosses this hyperplane, in that case traverse the other branch
	if max > math.Abs(point[axis]-node.v.point[axis]) {
		knn.nearestNieghbours(otherBranch, point, nearest)
	}
}

func maxDist(nearest []*distClassPair) (float64, int) {
	var max float64 = -1
	var maxIndex int
	for i, n := range nearest {
		if n == nil {
			return math.MaxFloat64, i
		}
		if n.dist > max {
			max = n.dist
			maxIndex = i
		}
	}
	return max, maxIndex
}

func insert(root *kdtree, values []value) {
	if len(values) == 1 {
		root.v = values[0]
		return
	}

	axis := root.depth % len(values[0].point)
	i := medianIndex(values, axis)

	leftValues := values[:i]
	pivot := values[i]
	rightValues := values[i+1:]

	root.v = pivot
	if len(leftValues) != 0 {
		root.left = &kdtree{depth: root.depth + 1}
		insert(root.left, leftValues)
	}
	if len(rightValues) != 0 {
		root.right = &kdtree{depth: root.depth + 1}
		insert(root.right, rightValues)
	}
}

func medianIndex(values []value, axis int) int {
	valuesInAxis := make([]float64, len(values))
	for i, val := range values {
		valuesInAxis[i] = val.point[axis]
	}
	sort.Float64s(valuesInAxis)
	mid := valuesInAxis[len(valuesInAxis)/2]
	for i, val := range values {
		if val.point[axis] > mid {
			return i
		}
	}
	return 0
}
