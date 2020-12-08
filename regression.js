var N, data, labels, drop_prob_val, activ;
var density = 5.0;  // defines the number the points for the function to be evaluated at.
// if it is small, then the curve will be more noisy and if it is high, it will be smooth/averaged.
var ss = 30.0;
var acc = 0.0;

var layer_defs, net, trainer, sum_y, sum_y_sq;

function restart_net(){
    // get dropout probability
    drop_prob_val = parseFloat($('#drop_prob').val());
    // get non-linearity
    if($("#nl_relu").is(':checked')){
        activ = 'relu';
    } else if($("#nl_sigmoid").is(':checked')) {
        activ = 'sigmoid';
    } else {
        activ = 'tanh';
    }
    layer_defs = [];
    layer_defs.push({type: 'input', out_sx: 1, out_sy:1, out_depth:1});
    layer_defs.push({type: 'dropout', drop_prob: drop_prob_val});
    layer_defs.push({type: 'fc', num_neurons: 15, activation: activ});
    layer_defs.push({type: 'dropout', drop_prob: drop_prob_val});
    layer_defs.push({type: 'fc', num_neurons: 25, activation: activ});
    layer_defs.push({type: 'regression', num_neurons: 1});

    net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    trainer= new convnetjs.SGDTrainer(net, {learning_rate: 0.01, momentum: 0.0, batch_size: 16, l2_decay: 1e-5});
    // console.log(WIDTH, HEIGHT)
    sum_y = Array();
    for (var x=0.0; x<=WIDTH; x+=density){
        sum_y.push(new cnnutil.Window(150, 0));     // caches 150 elements in one go and averages them
    }
    sum_y_sq = Array();
    for (var x=0.0; x<=WIDTH; x+=density){
        sum_y_sq.push(new cnnutil.Window(150, 0));
    }
    acc = 0.0;
}

function update_net(){
    // forward pass
    var netx = new convnetjs.Vol(1, 1, 1);  // creates a 3D voxel i.e or tensor because of restrictions from ConvNetJS
    avg_loss = 0.0;
    for (var i=0; i<100; i++){
        for (var ix=0; ix<N; ix++){
            netx.w = data[ix];
            stats = trainer.train(netx, labels[ix]);
            avg_loss += stats.loss;
        }
    }
    avg_loss /= i*N;
}

function regenerate_data(){
    // N = 20;
    N = parseInt($("#num_data").val());
    sum_y = Array();
    for (var x=0.0; x<=WIDTH; x+=density){
        sum_y.push(new cnnutil.Window(150, 0));
    }
    sum_y_sq = Array();
    for (var x=0.0; x<=WIDTH; x+=density){
        sum_y_sq.push(new cnnutil.Window(150, 0));
    }
    acc = 0.0;
    data = [];
    labels = [];
    for (var i=0; i<=N; i++){
        var x = Math.random()*10 - 5.0;
        if ($("#func_1").is(':checked')){
            var y = x*Math.sin(x);
        } else if ($("#func_2").is(':checked')){
            var y = Math.sin(2*x) - 2*Math.sin(x);
            // var y = 2*x;
        } else {
            var y = x**2*Math.cos(x)*Math.sin(x);
        }
        // var y = x*Math.cos(2*x)*Math.sin(x);
        data.push([x]);
        labels.push([y]);
    }
}

function myinit(){
    regenerate_data();
    restart_net();
}

function draw_reg(){
    ctx_reg.clearRect(0,0,WIDTH,HEIGHT);
    ctx_reg.fillStyle = "black";

    var netx = new convnetjs.Vol(1,1,1);

    ctx_reg.globalAlpha = 0.5;
    ctx_reg.beginPath();
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {

      netx.w[0] = (x-WIDTH/2)/ss;
      var a = net.forward(netx);
      var y = a.w[0];
      sum_y[c].add(y);
      sum_y_sq[c].add(y*y);

      if(x===0) ctx_reg.moveTo(x, -y*ss+HEIGHT/2);
      else ctx_reg.lineTo(x, -y*ss+HEIGHT/2);
      c += 1;
    }
    acc += 1;
    ctx_reg.stroke();
    ctx_reg.globalAlpha = 1.;

    // draw axes
    ctx_reg.beginPath();
    ctx_reg.strokeStyle = 'rgb(50,50,50)';
    ctx_reg.lineWidth = 1;
    ctx_reg.moveTo(0, HEIGHT/2);
    ctx_reg.lineTo(WIDTH, HEIGHT/2);
    ctx_reg.moveTo(WIDTH/2, 0);
    ctx_reg.lineTo(WIDTH/2, HEIGHT);
    ctx_reg.stroke();

    // draw datapoints. Draw support vectors larger
    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    ctx_reg.globalAlpha = 0.6;
    ctx_reg.lineWidth = 1;
    for(var i=0;i<N;i++) {
      drawCircle(data[i]*ss+WIDTH/2, -labels[i]*ss+HEIGHT/2, 4.0);
    }

    // Draw the mean plus minus 2 standard deviations
    ctx_reg.beginPath();
    ctx_reg.strokeStyle = 'rgb(0,0,250)';
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y[c].get_average();
      if(x===0) ctx_reg.moveTo(x, -mean*ss+HEIGHT/2);   // this is the main blue line around with uncertainty is drawn.
      else ctx_reg.lineTo(x, -mean*ss+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();
    // Draw the uncertainty
    ctx_reg.fillStyle = 'rgb(0,100,50)';
    ctx_reg.globalAlpha = 0.2;
    uncertainty_slabs = 3;
    // terms required for calculating the model precision tau.
    l2 = 0.005;     // l2 is in fact l^2 where l is the prior length scale
    // l2 = 1;     // l2 is in fact l^2 where l is the prior length scale
    drop_prob_val = parseFloat($('#drop_prob').val());
    wgt_decay = 1e-5;
    tau = (l2*(1 - drop_prob_val))/(2*N*wgt_decay);
    tau_inv = tau**(-1)
    // tau_inv = (2 * N * 0.00001) / (1 - 0.05) / l2;

    for(var i = 1; i <= uncertainty_slabs; i++) {
      ctx_reg.beginPath();
      var c = 0;
      var start = 0
      for(var x=0.0; x<=WIDTH; x+= density) {
        var mean = sum_y[c].get_average();
        // This is a miscalculation spotted in the blog's Github issues. In the blog, tau_inv is added the "variance"
          // of the estimated outputs, whereas here, it is added to the std_dev. The tau_inv should be inside the sqrt.
        // var std = Math.sqrt(sum_y_sq[c].get_average() - mean * mean) + tau_inv;
        var std = Math.sqrt(sum_y_sq[c].get_average() - (mean * mean)  + tau_inv);
        mean += 2*std  * i/uncertainty_slabs;
        if(x===0) {
            start = -mean*ss+HEIGHT/2;
            ctx_reg.moveTo(x, start);
        }
        else ctx_reg.lineTo(x, -mean*ss+HEIGHT/2);
        c += 1;
      }
      var c = sum_y.length - 1;
      for(var x=WIDTH; x>=0.0; x-= density) {
        var mean = sum_y[c].get_average();
        // var std = Math.sqrt(sum_y_sq[c].get_average() - mean * mean) + tau_inv;
        var std = Math.sqrt(sum_y_sq[c].get_average() - (mean * mean)  + tau_inv);
        mean -= 2*std * i/uncertainty_slabs;
        ctx_reg.lineTo(x, -mean*ss+HEIGHT/2);
        c -= 1;
      }
      ctx_reg.lineTo(0, start);
      ctx_reg.fill();
    }
    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    ctx_reg.globalAlpha = 1.;

    ctx_reg.fillStyle = "blue";
    ctx_reg.font = "bold 18px 'Times New Roman'";
    ctx_reg.fillText("average loss: " + avg_loss, 20, 20);
}

function mouseClick(x, y, shiftPressed){
    // adds a datapoint in the grid where the mouse is clicked
    // x = x / $(NPGcanvas).width() * WIDTH;
    // y = y / $(NPGcanvas).height() * HEIGHT;
    data.push([(x - WIDTH/2)/ss]);
    labels.push([-(y - HEIGHT/2)/ss]);
    N += 1;
}
