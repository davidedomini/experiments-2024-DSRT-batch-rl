package it.unibo.interop

import me.shadaj.scalapy.py

object PythonModules {
  val pythonUtils: py.Module = py.module("RLutils")
  val torch: py.Module = py.module("torch")
}
