/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


class DLAEngine[T: ClassTag]
(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  @native def dlaInference(s: String): Array[Long]
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    System.loadLibrary("DLAEngineImpl")
    System.loadLibrary("dla")
    System.loadLibrary("mongoose")
    val re = dlaInference("").toList
    println("DLA result: " + re(0) + " FPS")
    return Tensor[T](1, 2, 3)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    return Tensor[T](1, 2, 3)
  }
}

object DLAEngine {
  def getModel[@specialized(Float, Double) T: ClassTag](model : String)
      (implicit ev: TensorNumeric[T]) : DLAEngine[T] = {
    new DLAEngine[T]()
  }

  def setModelPath(modelPath : String) : Unit = {

  }

  def setImagePath(imagePath : String) : Unit = {

  }

  def setBatchSize(imagePath : Int) : Unit = {

  }
  def getImages[@specialized(Float, Double) T: ClassTag]
      (implicit ev: TensorNumeric[T]): Tensor[T] = {
    return Tensor[T](1, 2, 3)
  }
}

