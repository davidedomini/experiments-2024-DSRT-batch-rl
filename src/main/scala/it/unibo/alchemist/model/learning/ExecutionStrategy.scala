package it.unibo.alchemist.model.learning

import it.unibo.alchemist.model.Environment

trait ExecutionStrategy {
  def execute(environment: Environment[Any, Nothing]): Unit
}

trait GlobalExecution extends ExecutionStrategy {
  def execute(environment: Environment[Any, Nothing]): Unit
}

trait LocalExecution extends ExecutionStrategy {
  def execute(environment: Environment[Any, Nothing]): Unit
}