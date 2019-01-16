// You can edit this code!
// Click here and start typing.
package main

import (
        "fmt"
        "time"
        "math/rand"
        )

  var turn=0
  var length=0
  var width=0
  var turn_pat [4]int = [4]int{40,50,60,80}
  var width_pat [4]int = [4]int{8,9,10,12}
  var length_pat [4]int = [4]int{11,12,12,12}
  var field=make([][]int,12)

var test [4]int = [4]int{0,0,0,0}
var as=make([][]int,4)

func retPField(i int){
  if(i==0){ // 初期並びが横並び
    field=make([][]int,(length+1)/2)
    for i:=0; i<(length+1)/2; i++{
      field[i]=make([]int, width)
      for j:=0; j<width; j++ {
        field[i][j]=rand.Intn(32)-16
      }
    }

    tmp_field:=make([][]int,length/2)
    for i:=0; i<length/2; i++{
      tmp_field[i]=make([]int, width)
      tmp_field[i]=field[((length)/2)-1-i]
    }
    field=append(field,tmp_field...)
    /*
    for i:=0; i<length; i++{
      for j:=0; j<width; j++ {
        fmt.Println(field[i][j])
      }
      fmt.Println("\n")
    }
    */
  }else if(i==1){ // 初期並びが縦並び
    field=make([][]int,length)
    for i:=0; i<length; i++{
      field[i]=make([]int,width)
      for j:=0; j<width; j++ {
        if j<(width+1)/2{
          field[i][j]=rand.Intn(32)-16
        }else if j>=(width+1)/2{
          field[i][j]=field[i][width-(j+1)]
        }
      }
    }
    /*
    for i:=0; i<length; i++{
      for j:=0; j<width; j++ {
        fmt.Println(field[i][j])
      }
      fmt.Println("\n")
    }
    */
  }
}

func main() {
  /*
  rand.Seed(time.Now().UnixNano())
  turn_num:=-1 * rand.Intn(4)
  fmt.Println(turn_num)

  turn=turn_pat[turn_num]
  length=length_pat[turn_num]
  width=width_pat[turn_num]
  a:=rand.Intn(2)
  retPField(a)
  fmt.Println(a)
  fmt.Println(field)
  */

  for i:=0;i<4;i++{
    as[i]=make([]int,6)
  }
  for i:=0;i<4;i++{
    for j:=0;j<6;j++{
      rand.Seed(time.Now().UnixNano())
      as[i][j]=rand.Intn(4)
    }
  }
  fmt.Println(as)
  fmt.Println(as[0][3])
  
}
