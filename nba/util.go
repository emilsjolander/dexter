package nba

import "errors"

var (
	WrongDimensionError   = errors.New("Dimensionality of data does not match prior data")
	NoDataError           = errors.New("Cannot fit model without training data")
	NoClassificationError = errors.New("No Class was found for this data point")
)
