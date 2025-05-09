'''
// mongosh
// use database


db.users.insertOne({
  name: "Alice",
  age: 25,
  email: "alice@example.com"
})


db.users.find()


db.users.find({age: {$gte : 18}})


db.users.updateOne(
  {name: "Alice"},
  {$set: {age: 26}}
)


db.users.updateMany(
  {age: {$lt : 18}},
  {$set: {status: "minor"}}
)


db.users.deleteOne({
  name: "Alice"
})


db.users.deleteMany({age : {$lt : 13}})

db.students.insertMany(
  {roll_no: 101, name : "Alice", age : 20, grade : "A"},
  {roll_no: 102, name : "Bob", age : 21, grade : "B"},
  {roll_no: 103, name : "Charlie", age : 22, grade : "A"},
  {roll_no: 104, name : "Suman", age : 23, grade : "C"},
  {roll_no: 105, name : "Eve", age : 20, grade : "B"}

)


db.students.find()


db.students.updateOne(
  {roll_no : 105},
  {$set : {grade: "A"}}
)


db.students.updateMany(
  {age: {$lt : 22}},
  {$set : {grade : "B"}}
)


db.students.deleteOne({roll_no: 101})


db.students.deleteMany({grade : "C"})

'''
