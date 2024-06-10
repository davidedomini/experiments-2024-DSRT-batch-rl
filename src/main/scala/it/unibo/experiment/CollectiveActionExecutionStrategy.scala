package it.unibo.experiment

import it.unibo.alchemist.model.Environment
import it.unibo.alchemist.model.learning.GlobalExecution

import scala.jdk.CollectionConverters.IteratorHasAsScala

class CollectiveActionExecutionStrategy extends GlobalExecution {

  override def execute(environment: Environment[Any, Nothing]): Unit = {

  }


  private def nodes(environment: Environment[Any, Nothing]) =
    environment.getNodes.iterator().asScala.toList
}
