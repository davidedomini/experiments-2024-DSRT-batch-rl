package it.unibo.alchemist.model.learning

import it.unibo.alchemist.model.{Environment, Position}

trait ExecutionStrategy[T, P <: Position[P]] {
  def execute(environment: Environment[T, P]): Unit
}

trait GlobalExecution[T, P <: Position[P]] extends ExecutionStrategy[T, P] {
  def execute(environment: Environment[T, P]): Unit
}

trait LocalExecution[T, P <: Position[P]] extends ExecutionStrategy[T, P] {
  def execute(environment: Environment[T, P]): Unit
}