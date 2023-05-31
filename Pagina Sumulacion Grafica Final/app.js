let express = require('express');
let app = express();

// let path = require('path')
// let publicPath = path.join(__dirname,"public")
app.use(express.static("public"));
app.get('/',(req,res)=>{
    res.sendFile(`${publicPath}/index.html`)
})
app.listen(4000)