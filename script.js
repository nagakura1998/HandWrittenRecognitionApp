const canvas = document.getElementById("canvas");
const clearEl = document.getElementById("clear");
const recogniseEl = document.getElementById("recognise");
const solutionContainerEl = document.getElementById("solution-container");
const ctx = canvas.getContext("2d");
word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X', 34:'Y',35:'Z'}

const sessionOption = { executionProviders: ['wasm', 'webgl'] };
var inferenceSession;
async function createInferenceSession(onnxModelURL, sessionOption) 
{
    try {
        inferenceSession = await ort.InferenceSession.create(onnxModelURL, sessionOption);
    } catch (e) {
        console.log(`failed to load ONNX model: ${e}.`);
    }
}

async function InitializeModel(){
    const onnxModelURL = `./model.onnx`;
    await createInferenceSession(onnxModelURL, sessionOption);
}

InitializeModel();

let size = 15;
let isPressed = false;
let color = "black";
let x = undefined;
let y = undefined;
let isEraser = false;

canvas.addEventListener("mousedown", (e) => {
    isPressed = true;

    x = e.offsetX;
    y = e.offsetY;
});

canvas.addEventListener("mouseup", (e) => {
    isPressed = false;

    x = undefined;
    y = undefined;
});

canvas.addEventListener("mousemove", (e) => {
    if (isPressed) {
        if (!isEraser) {
            const x2 = e.offsetX;
            const y2 = e.offsetY;
            drawCircle(x2, y2);
            drawLine(x, y, x2, y2);
            x = x2;
            y = y2;
        }
        else{
            mouseX = e.pageX - canvas.offsetLeft;
            mouseY = e.pageY - canvas.offsetTop;
            ctx.clearRect(mouseX, mouseY, size, size);
        }
    }
});

function drawCircle(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = size * 2;
    ctx.stroke();
}

clearEl.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

function exportToJsonFile(jsonData) {
    let dataStr = JSON.stringify(jsonData);
    let dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

    let exportFileDefaultName = 'data.json';

    let linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}

function downloadCanvasImage(tmpCanvas){
    let downloadLink = document.createElement('a');
    downloadLink.setAttribute('download', 'CanvasAsImage.png');
    let dataURL = tmpCanvas.toDataURL('image/png');
    let url = dataURL.replace(/^data:image\/png/,'data:application/octet-stream');
    downloadLink.setAttribute('href', url);
    downloadLink.click();
}

async function RunPrediction(inputTensor)
{
    const feeds = { x: inputTensor };

    const results = await inferenceSession.run(feeds);
    
    resultData = results.dense_2.data;

    let result = resultData.indexOf(Math.max(...resultData));
    solutionContainerEl.innerText = word_dict[result]
}

recogniseEl.addEventListener("click",()=>{
    var tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    var tmpctx = tmpCanvas.getContext("2d");
    tmpctx.drawImage(canvas, 0,0,28,28);
    
    //downloadCanvasImage(tmpCanvas);
    //downloadCanvasImage(canvas);
    
    const imgData = tmpctx.getImageData(0, 0, 28, 28);
    const inputData = Float32Array.from(imgData.data)
    
    let tmpData = []
    for (let i = 0; i< inputData.length; i+=4){
        tmpData.push(inputData[i+3]);
    }
    
    const inputTensor = new ort.Tensor('float32', tmpData, [1, 28, 28]);
    
    //exportToJsonFile({data:inputData})

    RunPrediction(inputTensor);
    
})