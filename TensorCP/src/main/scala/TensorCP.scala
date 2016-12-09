/**
  * Created by cqwcy201101 on 12/6/16
  */
import org.apache.spark.mllib.linalg.distributed.CloudCP._

import breeze.linalg.{DenseVector => BDV}
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRowMatrix, IndexedRow}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.control.Breaks


object TensorCP {

  def main(args: Array[String]): Unit = {
    val inputFile = "/users/cqwcy201101/Desktop/test.md"
    val output = "/users/cqwcy201101/Desktop/result/"

    /*--------------------------------
    val inputm1 =  "/Users/cqwcy201101/Desktop/MA.md"
    val inputm2 =  "/Users/cqwcy201101/Desktop/MB.md"
    */

    val time_s:Double=System.nanoTime()



    val conf = new SparkConf().setAppName("TensorCP").setMaster("local").set("spark.ui.port","4040")
    val sc = new SparkContext(conf)
    val inputData = sc.textFile(inputFile)

    val TensorData = CloudCP.readFile(inputData).cache()
/*
    //-----------------------------
    val MAdata = sc.textFile(inputm1)
      .map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    val MBdata = sc.textFile(inputm2)
      .map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
    //----------------------------
*/
    val SizeVector = CloudCP.RDD_VtoRowMatrix(TensorData)
      .computeColumnSummaryStatistics().max


    val SizeA:Long = SizeVector.apply(0).toLong+1
    val SizeB:Long = SizeVector.apply(1).toLong+1
    val SizeC:Long = SizeVector.apply(2).toLong+1
    val R:Int = 2
    val Dim_1:Int = 0
    val Dim_2:Int = 1
    val Dim_3:Int = 2

    val iterat =100
    val tolerance = 0.0001



    var MA:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeA,R,sc)
    var MB:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeB,R,sc)
    var MC:IndexedRowMatrix = CloudCP.InitialIndexedRowMatrix(SizeC,R,sc)

    var N:Int = 0



/*
    var MA:IndexedRowMatrix = new IndexedRowMatrix(MAdata.zipWithIndex().map{case (x,y) => IndexedRow(y,x)})
    var MB:IndexedRowMatrix = new IndexedRowMatrix(MBdata.zipWithIndex().map{case (x,y) => IndexedRow(y,x)})
    var MC:IndexedRowMatrix = new IndexedRowMatrix(MAdata.zipWithIndex().map{case (x,y) => IndexedRow(y,x)})
*/

    var ATA = CloudCP.Compute_MTM_RowMatrix(MA)
    var BTB = CloudCP.Compute_MTM_RowMatrix(MB)
    var CTC = CloudCP.Compute_MTM_RowMatrix(MC)

    var Lambda:BDV[Double] = BDV.zeros(R)
    var fit:Double = 0.0
    var prev_fit:Double =0.0
    var val_fit:Double =0.0

    /*
        val testM1 = CalculateM1(TensorData,MB,MC,Dim_1,SizeA,R,sc)
        testM1.rows.collect().foreach(println)
        val testM2 = CalculateM2(MB,MC)
        println(testM2)
        val testM1M1 = testM1.multiply(testM2)
        testM1M1.rows.collect().foreach(println)

        Lambda = UpdateLambda(testM1M1)
        //val RDDLambda = sc.parallelize(Vector(Lambda))
        //var boradambda = sc.broadcast(RDDLambda.collect())

        println(Lambda)

        val testM1M1_2 = testM1M1.rows.map(x => (x.index, BDV[Double](x.vector.toArray))).mapValues(x=> (x :/ Lambda))

          //IndexedRow(x.index, Vectors.dense(((BDV[Double](x.vector.toArray)) :/ Lambda ).toArray)))
        testM1M1_2.collect().foreach(println)

        val testM1M2_3 = UpdateMatrix(TensorData,Lambda,MB,MC,Dim_1,SizeA,R,sc)
        testM1M2_3.rows.collect().foreach(println)
    */


    /*
        val testB1 = CalculateM1(TensorData,MC,MA,Dim_2,SizeB,R,sc)
        testB1.rows.collect().foreach(println)
        val testB2 = testB1.multiply(CalculateM2(MC,MA))
        testB2.rows.collect().foreach(println)

        for (i <- 0 until R){
          Lambda = UpdateLambda(testB2,i)
          MB = UpdateMatrix(TensorData,Lambda,MC,MA,Dim_2,SizeB,R,sc)

        }
        println(Lambda)
        MB.rows.collect().foreach(println)
    */



    val loop = new Breaks

    loop.breakable{

      for (i <- 0 until  iterat)
      {
        MA = CloudCP.UpdateMatrix(TensorData,MB,MC,Dim_1,SizeA,R,sc)
        Lambda = CloudCP.UpdateLambda(MA,i)
        MA = CloudCP.NormalizeMatrix(MA,Lambda)
        ATA = CloudCP.Compute_MTM_RowMatrix(MA)

        MB = CloudCP.UpdateMatrix(TensorData,MC,MA,Dim_2,SizeB,R,sc)
        Lambda =CloudCP.UpdateLambda(MB,i)
        MB = CloudCP.NormalizeMatrix(MB,Lambda)
        BTB = CloudCP.Compute_MTM_RowMatrix(MB)

        MC= CloudCP.UpdateMatrix(TensorData,MA,MB,Dim_3,SizeC,R,sc)
        Lambda = CloudCP.UpdateLambda(MC,i)
        MC = CloudCP.NormalizeMatrix(MC,Lambda)
        CTC= CloudCP.Compute_MTM_RowMatrix(MC)

        prev_fit = fit
        fit = CloudCP.ComputeFit(TensorData,Lambda,MA,MB,MC,ATA,BTB,CTC)
        val_fit = abs(fit - prev_fit)

        N = N +1

        if (val_fit < tolerance)
          loop.break
      }


    }

    //---------------
    val time_e:Double=System.nanoTime()

    //---------------


    /*
    MA.rows.collect().foreach(println)
    MB.rows.collect().foreach(println)
    MC.rows.collect().foreach(println)
    */



    CloudCP.OutputResult(MA,output+"MatrixA")
    CloudCP.OutputResult(MB,output+"MatrixB")
    CloudCP.OutputResult(MC,output+"MatrixC")

    
    println(val_fit)
    println(N)
    println(Lambda)

    println("Running time is:")
    println((time_e-time_s)/1000000000+"s\n")



    


  }

}
