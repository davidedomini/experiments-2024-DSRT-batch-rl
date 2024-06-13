package it.unibo.experiment

import it.unibo.alchemist.model.learning.{State, StateEncoder}

case class FlockState(
    myPosition: (Double, Double),
    neighborsPosition: Seq[(Double, Double)]
) extends State

object FlockState {
  implicit val stateEncoder: StateEncoder[FlockState] = (state: FlockState) => {
    val fill = List.fill(ExperimentParams.neighbors * ExperimentParams.neighborPositionSize)(0.0)
    (state.neighborsPosition.flatMap { case (l, r) => List(l, r) } ++ fill)
      .take(ExperimentParams.neighbors * ExperimentParams.neighborPositionSize)
  }
}
