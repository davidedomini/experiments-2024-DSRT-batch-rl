package it.unibo.alchemist.model.learning

import collection.mutable.ArrayDeque
import scala.collection.mutable
import scala.util.Random

case class Experience[S <: State](
                       actualState: S,
                       action: Action,
                       reward: Double,
                       nextState: S)
                     (implicit actionEncoder: ActionEncoder,
                      stateEncoder: StateEncoder[S]) {

  def encode: (Seq[Double], Int, Double, Seq[Double]) = {
    (stateEncoder.encode(actualState), actionEncoder.encode(action), reward, stateEncoder.encode(nextState))
  }

}

trait ExperienceBuffer[S <: State] {

  /** Inserts new experience */
  def insert(experience: Experience[S]): Unit

  /** Empty the buffer */
  def reset(): Unit

  /** Gets a sub-sample of the experience stored by the agents */
  def sample(batchSize: Int): Seq[Experience[S]]

  /** Gets all the experience stored by the agents */
  def getAll: Seq[Experience[S]]

  def addAll(experience: Seq[Experience[S]]): Unit

  /** Gets the buffer size */
  def size: Int

}

object ExperienceBuffer {
  def apply[S <: State](size: Int): ExperienceBuffer[S] = {
    new BoundedQueue(size, 42)
  }

  private class BoundedQueue[S <: State](bufferSize: Int, seed: Int) extends ExperienceBuffer[S] {

    private var queue: ArrayDeque[Experience[S]] = ArrayDeque.empty

    override def reset(): Unit = queue = ArrayDeque.empty[Experience[S]]

    override def insert(experience: Experience[S]): Unit =
      queue = (queue :+ experience).takeRight(bufferSize)

    override def sample(batchSize: Int): Seq[Experience[S]] =
      new Random(seed).shuffle(queue).take(batchSize).toSeq

    override def getAll: Seq[Experience[S]] = queue.toSeq

    override def addAll(experience: Seq[Experience[S]]): Unit =
      queue = queue :++ experience

    override def size: Int = queue.size
  }

}