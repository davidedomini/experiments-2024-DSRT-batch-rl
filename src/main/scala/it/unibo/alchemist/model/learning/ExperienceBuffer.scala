package it.unibo.alchemist.model.learning

import collection.mutable.ArrayDeque
import scala.collection.mutable
import scala.util.Random

case class Experience(
                       actualState: State,
                       action: Action,
                       reward: Double,
                       nextState: State)
                     (implicit actionEncoder: ActionEncoder,
                      stateEncoder: StateEncoder) {

  def encode: (Seq[Double], Int, Double, Seq[Double]) = {
    (stateEncoder.encode(actualState), actionEncoder.encode(action), reward, stateEncoder.encode(nextState))
  }

}

trait ExperienceBuffer {

  /** Inserts new experience */
  def insert(experience: Experience): Unit

  /** Empty the buffer */
  def reset(): Unit

  /** Gets a sub-sample of the experience stored by the agents */
  def sample(batchSize: Int): Seq[Experience]

  /** Gets all the experience stored by the agents */
  def getAll: Seq[Experience]

  /** Gets the buffer size */
  def size: Int

}

object ExperienceBuffer {
  def apply(size: Int): ExperienceBuffer = {
    new BoundedQueue(size, 42)
  }

  private class BoundedQueue(bufferSize: Int, seed: Int) extends ExperienceBuffer {

    private var queue: mutable.ArrayDeque[Experience] = mutable.ArrayDeque.empty

    override def reset(): Unit = queue = mutable.ArrayDeque.empty[Experience]

    override def insert(experience: Experience): Unit =
      queue = (queue :+ experience).takeRight(bufferSize)

    override def sample(batchSize: Int): Seq[Experience] =
      new Random(seed).shuffle(queue).take(batchSize).toSeq

    override def getAll: Seq[Experience] = queue.toSeq

    override def size: Int = queue.size
  }

}