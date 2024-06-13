package it.unibo.alchemist.model.learning

import it.unibo.alchemist.model.{Environment, Position}
import org.apache.commons.math3.random.RandomGenerator

trait ExecutionStrategy[T, P <: Position[P]] {
  var round: Int = 0
  def execute(environment: Environment[T, P], randomGenerator: RandomGenerator): Unit

  def progressRound(): Unit = round += 1
}

trait GlobalExecution[T, P <: Position[P]] extends ExecutionStrategy[T, P] {
  def execute(environment: Environment[T, P], randomGenerator: RandomGenerator): Unit
}

trait LocalExecution[T, P <: Position[P]] extends ExecutionStrategy[T, P] {
  def execute(environment: Environment[T, P], randomGenerator: RandomGenerator): Unit
}
