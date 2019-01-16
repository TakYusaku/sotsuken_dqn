// learning only port num is 8000
// 今まで通り完全全てランダム
package main

import (
    "fmt"
    "net/http"
    "log"
    "math/rand"
    "time"
    "strconv"
    "strings"
)
type String string
// http.HandleFuncに登録する関数
// http.ResponseWriterとhttp.Requestを受ける
var user=make([][]int,12)
var pcalc=make([][]int,12)
var field=make([][]int,12)
var turn=0
var length=0
var width=0
var pattern=0
var p=make(map[int]map[string]int)
var pcount [5]int = [5]int{0, 0, 0, 0, 0}
var init_order [4]int = [4]int{0, 0, 0, 0}
/*
var turn_pat [4]int = [4]int{40,50,60,80}
var width_pat [4]int = [4]int{8,9,10,12}
var length_pat [4]int = [4]int{11,12,12,12}
*/

func retPField(i int){  // 初期ならびによるポイントフィールドの作成
  if(i==0){ // 初期並びが横並び
    field=make([][]int,(length+1)/2)
    for i:=0; i<(length+1)/2; i++{
      field[i]=make([]int, width)
      for j:=0; j<width; j++ {
        rand.Seed(time.Now().UnixNano())
        a:=rand.Intn(99)+1
        if a <= 5{
          rand.Seed(time.Now().UnixNano())
          field[i][j]= -1 * rand.Intn(16)
        }else {
          rand.Seed(time.Now().UnixNano())
          field[i][j]=rand.Intn(15)+1
        }
      }
    }

    tmp_field:=make([][]int,length/2)
    for i:=0; i<length/2; i++{
      tmp_field[i]=make([]int, width)
      tmp_field[i]=field[((length)/2)-1-i]
    }
    field=append(field,tmp_field...)

  }else if(i==1){ // 初期並びが縦並び
    field=make([][]int,length)
    for i:=0; i<length; i++{
      field[i]=make([]int,width)
      for j:=0; j<width; j++ {
        if j<(width+1)/2{
          a:=rand.Intn(99)+1
          if a <= 5{
            rand.Seed(time.Now().UnixNano())
            field[i][j]= -1 * rand.Intn(16)
          }else {
            field[i][j]=rand.Intn(15)+1
          }
        }else if j>=(width+1)/2{
          field[i][j]=field[i][width-(j+1)]
        }
      }
    }
  }
}


func StartServer(w http.ResponseWriter, r *http.Request) {
    r.ParseForm()
    prov:=r.Form["init_order"]
    pat:=r.Form["pattern"]
    for i:=0; i<4; i++{
      init_order[i], _ =strconv.Atoi(prov[i])
    }
    pattern, _ =strconv.Atoi(pat[0])
    if pattern == 2 {
      rand.Seed(time.Now().UnixNano())
      pattern = rand.Intn(2)
    }
    // ターン数,縦横の選定
    rand.Seed(time.Now().UnixNano())
    turn=rand.Intn(40)+40
    length=rand.Intn(4)+8
    width=rand.Intn(4)+8

    fmt.Fprintf(w,"%d\n",turn)
    fmt.Fprintf(w,"%d\n",length)
    fmt.Fprintf(w,"%d\n",width)

    // ここからポイントフィールド作成
    retPField(pattern)
    for i:=0; i<length; i++{
      for j:=0; j<width; j++ {
        fmt.Fprintf(w,"%d ",field[i][j])
      }
      fmt.Fprintf(w,"\n")
    }
    // ここまで

    // ここからユーザーフィールド作成
    for i:=0; i<length; i++{
      user[i]=make([]int, width)
    }

    for i:=1; i<5; i++{
      p[i]=make(map[string]int)
    }
    x:=rand.Intn((width/2-1)-2)+1
    y:=rand.Intn((length/2-1)-2)+1
    p[1]["x"]=x
    p[1]["y"]=y
    p[2]["x"]=x
    p[2]["y"]=width-y-1
    p[3]["x"]=length-x-1
    p[3]["y"]=y
    p[4]["x"]=length-x-1
    p[4]["y"]=width-y-1

    for i:=1; i<5; i++{
      user[p[i]["x"]][p[i]["y"]]=init_order[i-1]
    }

    for i:=0; i<length; i++{
      for j:=0; j<width; j++ {
        fmt.Fprintf(w,"%d ",user[i][j])
      }
      fmt.Fprintf(w,"\n")
    }

}

func MoveServer(w http.ResponseWriter, r *http.Request) {
    // fmt.Fprintf(w, "move\n") yusaku
    r.ParseForm()
    //curl -X POST localhost:8000/move -d "usr=1&d=right"
    u,_:=strconv.Atoi(r.FormValue("usr"))
    fmt.Println(u)
    fmt.Println(r.FormValue("d"))
    //d:=r.FormValue("d")
    d:=strings.Split(r.FormValue("d"), "")
    if(d[0]=="z"){
      pcount[u]++
      return
    }
    /*
    for i:=0; i<len(d); i++{
      if d[i]=="r"{p[u]["y"]++
      }else if d[i]=="l"{p[u]["y"]--
      }else if d[i]=="u"{p[u]["x"]--
      }else if d[i]=="d"{p[u]["x"]++}
    }
    */
    tmp_px:=p[init_order[u-1]]["x"]
    tmp_py:=p[init_order[u-1]]["y"]
    for i:=0; i<len(d); i++{
      if d[i]=="r"{tmp_py++
      }else if d[i]=="l"{tmp_py--
      }else if d[i]=="u"{tmp_px--
      }else if d[i]=="d"{tmp_px++}
    }
    if 0<=tmp_px && tmp_px<length && 0<=tmp_py && tmp_py<width {
      if u==1||u==2 {
        if user[tmp_px][tmp_py]==0 || user[tmp_px][tmp_py]==5 {
          user[p[init_order[u-1]]["x"]][p[init_order[u-1]]["y"]]=5
        }else{
          fmt.Fprintf(w,"is_panel \n")  // ;;;
          return
        }
      }else{
        if user[tmp_px][tmp_py]==0 || user[tmp_px][tmp_py]==6 {
          user[p[init_order[u-1]]["x"]][p[init_order[u-1]]["y"]]=6
        }else{
          fmt.Fprintf(w,"is_panel \n")  // ;;;
          return
        }
      }
      p[init_order[u-1]]["x"]=tmp_px
      p[init_order[u-1]]["y"]=tmp_py
    }else{  // out of field
      fmt.Fprintf(w,"Error \n")  // ;;;
      return
    }
    user[p[init_order[u-1]]["x"]][p[init_order[u-1]]["y"]]=u
    pcount[u]++
    if(pcount[1]==pcount[2]&&pcount[2]==pcount[3]&&pcount[3]==pcount[4]){
      pcount[0]=pcount[1]
      fmt.Fprintf(w,"%d ",pcount[0])
    }
    if(turn==pcount[0]){
      fmt.Fprintf(w,"end the game \n")
    }
}

func RemoveServer(w http.ResponseWriter, r *http.Request) {
  // fmt.Fprintf(w, "remove\n") yusaku
  r.ParseForm()
  //curl -X POST localhost:8000/move -d "usr=1&d=right"
  u,_:=strconv.Atoi(r.FormValue("usr"))
  fmt.Println(u)
  fmt.Println(r.FormValue("d"))
  d:=strings.Split(r.FormValue("d"), "")
  tmp_px:=p[init_order[u-1]]["x"]
  tmp_py:=p[init_order[u-1]]["y"]
  for i:=0; i<len(d); i++{
    if d[i]=="r"{tmp_py++
    }else if d[i]=="l"{tmp_py--
    }else if d[i]=="u"{tmp_px--
    }else if d[i]=="d"{tmp_px++}
  }
  if 0<=tmp_px && tmp_px<length && 0<=tmp_py && tmp_py<width {
    if user[tmp_px][tmp_py]!=1&&user[tmp_px][tmp_py]!=2&&user[tmp_px][tmp_py]!=3&&user[tmp_px][tmp_py]!=4 {user[tmp_px][tmp_py]=0}
  }else{
    fmt.Fprintf(w,"Error \n")
    return
  }

  pcount[u]++
  if(pcount[1]==pcount[2]&&pcount[2]==pcount[3]&&pcount[3]==pcount[4]){
    pcount[0]=pcount[1]
    fmt.Fprintf(w,"%d ",pcount[0])
  }
  if(turn==pcount[0]){
    fmt.Fprintf(w,"end the game \n")
  }
}

func ShowServer(w http.ResponseWriter, r *http.Request) {
  for i:=0; i<length; i++{
    for j:=0; j<width; j++ {
      fmt.Fprintf(w,"%d ",field[i][j])
    }
    fmt.Fprintf(w,"\n")
  }
  for i:=0; i<length; i++{
    for j:=0; j<width; j++ {
      fmt.Fprintf(w,"%d ",user[i][j])
    }
    fmt.Fprintf(w,"\n")
  }
}

func UsrpointServer(w http.ResponseWriter, r *http.Request) {
  // fmt.Fprintf(w, "usrpoint\n") yusaku
  r.ParseForm()
  u,_:=strconv.Atoi(r.FormValue("usr"))
  fmt.Println(p[init_order[u-1]]["x"])
  fmt.Println(p[init_order[u-1]]["y"])
  fmt.Fprintf(w,"%d ",p[init_order[u-1]]["y"])
  fmt.Fprintf(w,"%d",p[init_order[u-1]]["x"])
}

func myAbs(x int) int{
  if(x<0){return -x}
  return x
}

var use5[60][60] bool
var use6[60][60] bool
var came[60][60] bool
var dx [4]int = [4]int{1, 0, -1, 0}
var dy [4]int = [4]int{0, 1, 0, -1}
var flag bool
var cnt int
func check_area(y int,x int ,wall int)bool{
  cnt++
  if(cnt>=width*length*2){return true}
  ret:=true
  came[y][x]=true
  if(!flag){return false}
  if(pcalc[y][x]==wall){return true}
  for i:=0;i<4;i++{
    nx:=x+dx[i]
    ny:=y+dy[i]
    tmp:=true
    if(nx<0||ny<0||nx>=width||ny>=length){
      flag=false
      return false
    }
    if(!came[ny][nx]){tmp=check_area(ny,nx,wall)}
    if(!tmp){ret=false}
  }
  return ret
}

func init_check_area(){
  flag=true
  cnt=0
  for i:=0;i<length;i++{
    for j:=0;j<width;j++{
      came[i][j]=false
    }
  }
}

func JudgeServer(w http.ResponseWriter, r *http.Request) { // ;;;
    // fmt.Fprintf(w, "move\n") yusak
    // curl -X POST localhost:8000/judgedirection -d "usr=1&d=r"
    r.ParseForm()
    //curl -X POST localhost:8000/move -d "usr=1&d=right"
    u,_:=strconv.Atoi(r.FormValue("usr"))
    fmt.Println(u)
    fmt.Println(r.FormValue("d"))
    //d:=r.FormValue("d")
    d:=strings.Split(r.FormValue("d"), "")

    tmp_px:=p[init_order[u-1]]["x"]
    tmp_py:=p[init_order[u-1]]["y"]
    for i:=0; i<len(d); i++{
      if d[i]=="r"{tmp_py++
      }else if d[i]=="l"{tmp_py--
      }else if d[i]=="u"{tmp_px--
      }else if d[i]=="d"{tmp_px++}
    }
    if 0<=tmp_px && tmp_px<length && 0<=tmp_py && tmp_py<width {
      if u==1||u==2 {
        fmt.Fprintf(w,"%d ",tmp_py)  // ;;;
        fmt.Fprintf(w,"%d",tmp_px)  // ;;;
        fmt.Fprintf(w,"\n") // ;;;
        if user[tmp_px][tmp_py]==0 || user[tmp_px][tmp_py]==5 {
          fmt.Fprintf(w,"OK \n")
        }else{
          fmt.Fprintf(w,"is_panel \n")  // ;;;
          return
        }
      }else{
        fmt.Fprintf(w,"%d ",tmp_py)  // ;;;
        fmt.Fprintf(w,"%d",tmp_px)  // ;;;
        fmt.Fprintf(w,"\n")  // ;;;
        if user[tmp_px][tmp_py]==0 || user[tmp_px][tmp_py]==6 {
          fmt.Fprintf(w,"OK \n")
        }else{
          fmt.Fprintf(w,"is_panel \n")  // ;;;
          return
        }
      }
      // p[u]["x"]=tmp_px
      // p[u]["y"]=tmp_py
    }else{  // out of field
      fmt.Fprintf(w,"%d ",p[init_order[u-1]]["y"])  // ;;;
      fmt.Fprintf(w,"%d",p[init_order[u-1]]["x"])  // ;;;
      fmt.Fprintf(w,"\n") // ;;;
      fmt.Fprintf(w,"Error \n")  // ;;;
      return
    }
    // user[p[u]["x"]][p[u]["y"]]=u
}


func PointcalcServer(w http.ResponseWriter, r *http.Request) {
  pcalc=user
  point5:=0
  point6:=0
  var field_point5=0
  var field_point6=0
  var tile_point5=0
  var tile_point6=0

  for i:=0; i<length; i++{
    for j:=0; j<width; j++ {
      if(pcalc[i][j]==1||pcalc[i][j]==2){
        pcalc[i][j]=5
      }
      if(pcalc[i][j]==3||pcalc[i][j]==4){
        pcalc[i][j]=6
      }
      //fmt.Fprintf(w,"%d ",pcalc[i][j])
    }
    //fmt.Fprintf(w,"\n")
  }

/*  fmt.Fprintf(w,"盤面\n")
  for i:=0; i<length; i++{
    for j:=0; j<width; j++ {
    fmt.Fprintf(w,"%04d ",field[i][j])
    }
    fmt.Fprintf(w,"\n")
  }*/

  //////////以上プリントでバッグ
  for i:=0;i<length;i++{
    for j:=0;j<width;j++{
      use5[i][j]=false
      use6[i][j]=false
    }
  }
  for i:=0;i<length;i++{
    for j:=0;j<width;j++{
      init_check_area() //flag=trueにして、cameをすべてfalseにする
      if(check_area(i,j,5)&&!use5[i][j]){
        use5[i][j]=true;

      }
      init_check_area()
      if(check_area(i,j,6)&&!use6[i][j]){
        use6[i][j]=true;

      }
    }
  }/*
  for y:=0;y<length;y++{//縦
    for x:=0;x<width;x++{//横
      if(use5[y][x]){
        if(pcalc[y][x]==5){point5+=field[y][x]
        }else{point5+=myAbs(field[y][x])}
      }
      if(use6[y][x]){
        if(pcalc[y][x]==6){point6+=field[y][x]
        }else{point6+=myAbs(field[y][x])}
      }

    }
  }*/
   // tile and field point
  for y:=0;y<length;y++{//縦
    for x:=0;x<width;x++{//横
      if(use5[y][x]){
        if(pcalc[y][x]==5){point5+=field[y][x]
        }
      }
      if(use6[y][x]){
        if(pcalc[y][x]==6){point6+=field[y][x]
        }
      }

    }
  }

  tile_point5 = point5
  tile_point6 = point6

  for y:=0;y<length;y++{//縦
    for x:=0;x<width;x++{//横
      if(use5[y][x]){
        if(pcalc[y][x]==5){
        }else{point5+=myAbs(field[y][x])}
      }
      if(use6[y][x]){
        if(pcalc[y][x]==6){
        }else{point6+=myAbs(field[y][x])}
      }

    }
  }

  field_point5 = point5 - tile_point5
  field_point6 = point6 - tile_point6

  fmt.Fprintf(w,"%d \n",tile_point5)
  fmt.Fprintf(w,"%d \n",field_point5)
  fmt.Fprintf(w,"%d \n",point5)
  fmt.Fprintf(w,"%d \n",tile_point6)
  fmt.Fprintf(w,"%d \n",field_point6)
  fmt.Fprintf(w,"%d \n",point6)

}


func main() {
    // http.HandleFuncにルーティングと処理する関数を登録
    http.HandleFunc("/start", StartServer)
    http.HandleFunc("/move", MoveServer)
    http.HandleFunc("/remove", RemoveServer)
    http.HandleFunc("/show", ShowServer)
    http.HandleFunc("/usrpoint", UsrpointServer)
    http.HandleFunc("/pointcalc", PointcalcServer)
    http.HandleFunc("/judgedirection", JudgeServer)
    // http.HandleFunc("/initpos", InitposServer)

    // ログ出力
    log.Printf("Start Go HTTP Server (port number is 8000,learning only)")

    // http.ListenAndServeで待ち受けるportを指定
    err := http.ListenAndServe(":8000", nil)

    // エラー処理
    if err != nil {
       log.Fatal("ListenAndServe: ", err)
    }
}
