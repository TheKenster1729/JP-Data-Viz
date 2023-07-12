const canvas = document.getElementById("canvas");
const ctx    = canvas.getContext("2d");

// rectangle
ctx.save(); // save default context

ctx.fillStyle = "#ff0000";
ctx.strokeStyle = "blue";
ctx.lineWidth = 10;
 
ctx.fillRect(50,50,50,50);
 
ctx.globalAlpha = 0.5;
ctx.strokeRect(50,50,50,50);
 
// dashed shape
ctx.restore();
ctx.save();

ctx.strokeStyle = "blue";
ctx.lineWidth = 2;
ctx.shadowBlur = 6;
ctx.shadowColor = "green";
ctx.shadowOffsetX = ctx.shadowOffsetY = 5;
ctx.setLineDash([5,2,1,2]);
 
ctx.beginPath();
ctx.moveTo(150,200);
ctx.lineTo(150,150);
ctx.lineTo(100,150);
ctx.bezierCurveTo(100,200,150,250,200,250);
ctx.lineTo(200,200);
ctx.closePath();
ctx.stroke();
 
ctx.restore();
ctx.save();
 
// quarter-circle
ctx.translate(100,250);
ctx.scale(0.5, 0.5);
ctx.strokeStyle = "red";
ctx.lineWidth = 4;
ctx.globalAlpha = 0.5;
 
ctx.beginPath();
ctx.moveTo(0,0);
ctx.lineTo(0,-100);
ctx.arcTo(-100,-100,-100,0,100);
ctx.lineTo(0,0);
ctx.stroke();
 
ctx.globalAlpha = 0.2;

ctx.beginPath();
ctx.arc(0,0,100,3.14,-1.57,false);
ctx.lineTo(0,0);
ctx.closePath();
ctx.fill();

ctx.restore();
ctx.save();
 
// text and half-circle
const text = "Canvas"
ctx.translate(250,150);
ctx.font = "24px monospace";
const textWidth = ctx.measureText(text).width;
const gradient = ctx.createLinearGradient(-50,-50,-50 + textWidth,-50);
gradient.addColorStop(0,"magenta");
gradient.addColorStop(1, "yellow");
 
ctx.fillStyle = gradient;
ctx.shadowColor = "transparent";

ctx.fillText(text, -45, -5);

ctx.scale(1.1, 1.1)
ctx.rotate(3.14);

ctx.beginPath();
ctx.arc(0,0,40,3.14,0,false);
ctx.fill();
 
ctx.restore();
ctx.save();
 
// image and clip
ctx.beginPath();
ctx.arc(175,75,40,0,6.28,false);
ctx.clip();

const image = new Image(100,100);
image.onload = function() {
    ctx.globalAlpha = 0.75;
    ctx.drawImage(this, 125, 25, this.width, this.height);
}
image.src = "test.jpg";
ctx.save();