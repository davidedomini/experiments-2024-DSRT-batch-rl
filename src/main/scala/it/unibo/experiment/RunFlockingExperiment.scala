package it.unibo.experiment

import it.unibo.alchemist.model.learning.BatchLearning

object RunFlockingExperiment extends App {

  private def strategies = List(
    new StateEvaluationStrategy[Any, Nothing](true),
    new ActionChoiceStrategy[Any, Nothing],
    new CollectiveActionExecutionStrategy[Any, Nothing],
    new StateEvaluationStrategy[Any, Nothing](false),
    new RewardEvaluationStrategy[Any, Nothing],
    new ExperienceCollectionStrategy[Any, Nothing]
  )

  private val learner = BatchLearning(
    strategies,
    "src/main/yaml/simulation.yml",
    List("seed"),
    globalRounds = 10,
    parallelism = 1,
    miniBatchSize = 64,
    seedName = "seed"
  )
  val time = System.currentTimeMillis()
  learner.startLearning()
  val computingTime = System.currentTimeMillis() - time
  private val learner2 = BatchLearning(
    strategies,
    "src/main/yaml/simulation2.yml",
    List("seed"),
    globalRounds = 10,
    parallelism = 2,
    miniBatchSize = 64,
    seedName = "seed"
  )

  val time2 = System.currentTimeMillis()
  learner2.startLearning()
  val computingTime2 = System.currentTimeMillis() - time2
  private val learner4 = BatchLearning(
    strategies,
    "src/main/yaml/simulation4.yml",
    List("seed"),
    globalRounds = 10,
    parallelism = 2,
    miniBatchSize = 64,
    seedName = "seed"
  )

  val time4 = System.currentTimeMillis()
  learner4.startLearning()
  val computingTime4 = System.currentTimeMillis() - time4

  private val learnerLong = BatchLearning(
    strategies,
    "src/main/yaml/simulation.yml",
    List("seed"),
    globalRounds = 40,
    parallelism = 1,
    miniBatchSize = 64,
    seedName = "seed"
  )
  val timeLong = System.currentTimeMillis()
  learnerLong.startLearning()

  val content = s"1,$computingTime\n2,$computingTime2\n4,$computingTime4"

  println(content)
}
