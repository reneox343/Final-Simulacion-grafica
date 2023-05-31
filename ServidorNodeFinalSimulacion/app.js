const { exec } = require('child_process');
const path = require('path');
let publicPath = path.join(__dirname,'Public')

exec(`python ${publicPath}\\predicter.py`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error}`);
      return;
    }
    console.log(`Python script output: ${stdout}`);

    let result = stdout.match(/-?[0-9]/g);
    result.forEach(function printing(element, index) {
      console.log(element);
  })
});
