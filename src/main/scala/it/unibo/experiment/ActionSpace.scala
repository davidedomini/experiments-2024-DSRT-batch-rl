package it.unibo.experiment

import it.unibo.alchemist.model.learning.{Action, ActionEncoder}

object ActionSpace {

  case object North extends Action
  case object South extends Action
  case object West extends Action
  case object East extends Action
  case object NorthEast extends Action
  case object SouthEast extends Action
  case object NorthWest extends Action
  case object SouthWest extends Action

  def all: Seq[Action] = Seq(North, South, West, East, NorthWest, SouthWest, NorthEast, SouthEast)

  implicit val encoder: ActionEncoder = (action: Action) => {
    all.indexOf(action)
  }

}
