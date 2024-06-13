package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.learning.ExecutionStrategy
import it.unibo.alchemist.model.{Action, Actionable, Condition, Dependency, Environment, GlobalReaction, Node, Position, Time, TimeDistribution}
import org.apache.commons.math3.random.RandomGenerator
import org.danilopianini.util.{ListSet, ListSets}

import java.util
import scala.jdk.CollectionConverters.IteratorHasAsScala

class GlobalReactionStrategyExecutor[T, P <: Position[P]](
    val environment: Environment[T, P],
    val randomGenerator: RandomGenerator,
    distribution: TimeDistribution[T],
    strategy: ExecutionStrategy[T, P]
) extends GlobalReaction[T] {

  private val actions: util.List[Action[T]] = util.List.of()
  private val conditions: util.List[Condition[T]] = util.List.of()

  override def getActions: util.List[Action[T]] = actions

  override def setActions(list: util.List[_ <: Action[T]]): Unit = {
    actions.clear()
    actions.addAll(list)
  }

  override def setConditions(list: util.List[_ <: Condition[T]]): Unit = {
    conditions.clear()
    conditions.addAll(list)
  }

  override def execute(): Unit = {
    executeBeforeUpdateDistribution()
    distribution.update(getTimeDistribution.getNextOccurence, true, getRate, environment)
  }

  private def executeBeforeUpdateDistribution(): Unit =
    strategy.execute(environment, randomGenerator) // TODO - check if correct

  override def getConditions: util.List[Condition[T]] = conditions

  override def getInboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()

  override def getOutboundDependencies: ListSet[_ <: Dependency] = ListSets.emptyListSet()

  override def getTimeDistribution: TimeDistribution[T] = distribution

  override def canExecute: Boolean = true

  override def initializationComplete(time: Time, environment: Environment[T, _]): Unit = {}

  override def update(time: Time, b: Boolean, environment: Environment[T, _]): Unit = {}

  override def compareTo(o: Actionable[T]): Int = getTau.compareTo(o.getTau)

  override def getRate: Double = distribution.getRate

  override def getTau: Time = distribution.getNextOccurence

  def nodes: List[Node[T]] = environment.getNodes.iterator().asScala.toList

}
