/*********************                                                        */
/*! \file nnet.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Kyle Julian\endverbatim
 **/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "nnet.h"

//Take in a .nnet filename with path and load the network from the file
//Inputs:  filename - const char* that specifies the name and path of file
//Outputs: void *   - points to the loaded neural network
NNet *load_network(const char* filename)
{
    //Load file and check if it exists
    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL)
    {
        return NULL;
    }

    //Initialize variables
    int bufferSize = 10240;
    char *buffer = new char[bufferSize];
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;
    NNet *nnet = new NNet();

    //Read int parameters of neural network
    line=fgets(buffer,bufferSize,fstream);
    while (strstr(line, "//")!=NULL)
        line=fgets(buffer,bufferSize,fstream); //skip header lines
    record = strtok(line,",\n");
    nnet->numLayers    = atoi(record);
    nnet->inputSize    = atoi(strtok(NULL,",\n"));
    nnet->outputSize   = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    //Allocate space for and read values of the array members of the network
    nnet->layerSizes = new int[(((nnet->numLayers)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->numLayers)+1); i++)
    {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }

    //Skip the unsused paramter
    line = fgets(buffer,bufferSize,fstream);

    //Load Min and Max values of inputs
    nnet->mins = new double[(nnet->inputSize)];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<(nnet->inputSize); i++)
    {
        nnet->mins[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->maxes = new double[(nnet->inputSize)];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<(nnet->inputSize); i++)
    {
        nnet->maxes[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    //Load Mean and Range of inputs
    nnet->means = new double[(((nnet->inputSize)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->inputSize)+1); i++)
    {
        nnet->means[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->ranges = new double[(((nnet->inputSize)+1))];
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0; i<((nnet->inputSize)+1); i++)
    {
        nnet->ranges[i] = atof(record);
        record = strtok(NULL,",\n");
    }

    //Allocate space for matrix of Neural Network
    //
    //The first dimension will be the layer number
    //The second dimension will be 0 for weights, 1 for biases
    //The third dimension will be the number of neurons in that layer
    //The fourth dimension will be the number of inputs to that layer
    //
    //Note that the bias array will have only number per neuron, so
    //    its fourth dimension will always be one
    //
    nnet->matrix = new double ***[((nnet->numLayers))];
    for (layer = 0; layer<(nnet->numLayers); layer++)
    {
        nnet->matrix[layer] = new double**[2];
        nnet->matrix[layer][0] = new double*[nnet->layerSizes[layer+1]];
        nnet->matrix[layer][1] = new double*[nnet->layerSizes[layer+1]];
        for (row = 0; row<nnet->layerSizes[layer+1]; row++)
        {
            nnet->matrix[layer][0][row] = new double[nnet->layerSizes[layer]];
            nnet->matrix[layer][1][row] = new double[1];
        }
    }

    //Iteration parameters

    layer = 0;
    param = 0;
    i=0;
    j=0;

    //Read in parameters and put them in the matrix
    while((line=fgets(buffer,bufferSize,fstream))!=NULL)
    {
        if(i>=nnet->layerSizes[layer+1])
        {
            if (param==0)
            {
                param = 1;
            }
            else
            {
                param = 0;
                layer++;
            }
            i=0;
            j=0;
        }
        record = strtok(line,",\n");
        while(record != NULL)
        {
            nnet->matrix[layer][param][i][j++] = atof(record);
            record = strtok(NULL,",\n");
        }
        j=0;
        i++;
    }
    nnet->inputs = new double[nnet->maxLayerSize];
    nnet->temp = new double[nnet->maxLayerSize];


    delete[] buffer;

    //return a pointer to the neural network
    return nnet;
}

//Deallocate memory used by a neural network
//Inputs:  void *network: Points to network struct
//Output:  void
void destroy_network(void *network)
{
    NNet *nnet = static_cast<NNet*>(network);
    int i=0, row=0;
    if (nnet!=NULL)
    {
      // NNet *nnet = static_cast<NNet*>(network);
        for(i=0; i<(nnet->numLayers); i++)
        {
            for(row=0;row<nnet->layerSizes[i+1];row++)
            {
                //free weight and bias arrays
                delete[] nnet->matrix[i][0][row];
                delete[] nnet->matrix[i][1][row];
            }

            //free pointer to weights and biases
            delete[](nnet->matrix[i][0]);
            delete[](nnet->matrix[i][1]);

            //free pointer to the layer of the network
            delete[](nnet->matrix[i]);
        }

        //free network parameters and the struct
        delete[](nnet->layerSizes);
        delete[](nnet->mins);
        delete[](nnet->maxes);
        delete[](nnet->means);
        delete[](nnet->ranges);
        delete[](nnet->matrix);
        delete[](nnet->inputs);
        delete[](nnet->temp);
        delete(nnet);
    }
}


//Complete one forward pass for a given set of inputs and return Q values
//Inputs:  void *network - pointer to the neural net struct
//         double *input - double pointer to the inputs to the network
//                             The inputs should be in form [r,th,psi,vOwn,vInt,tau,pa]
//                             with the angles being in radians
//         double *output - double pointer to the outputs from the network

//Output:  int - 1 if the forward pass was successful, -1 otherwise
int evaluate_network(void *network, double *input, double *output, bool normalizeInput, bool normalizeOutput)
{
    int i,j,layer;
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }

    //Cast void* to NNet struct pointer
    NNet *nnet = static_cast<NNet*>(network);
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    double ****matrix = nnet->matrix;

    //Normalize inputs

    if ( normalizeInput )
    {
        for (i=0; i<inputSize;i++)
        {
            if (input[i]>nnet->maxes[i])
            {
                nnet->inputs[i] = (nnet->maxes[i]-nnet->means[i])/(nnet->ranges[i]);
            }
            else if (input[i]<nnet->mins[i])
            {
                nnet->inputs[i] = (nnet->mins[i]-nnet->means[i])/(nnet->ranges[i]);
            }
            else
            {
                nnet->inputs[i] = (input[i]-nnet->means[i])/(nnet->ranges[i]);
            }
        }
    }
    else
    {
        for (i=0; i<inputSize;i++)
            nnet->inputs[i] = input[i];
    }

    double tempVal;
    // printf("number of layers = %u\n", numLayers);

    for (layer = 0; layer<(numLayers); layer++)
    {
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            double **weights = matrix[layer][0];
            double **biases  = matrix[layer][1];
            tempVal = 0.0;

            //Perform weighted summation of inputs
            for (j=0; j<nnet->layerSizes[layer]; j++)
            {
                tempVal += nnet->inputs[j]*weights[i][j];

            }

            //Add bias to weighted sum
            tempVal += biases[i][0];

            //Perform ReLU
            if (tempVal<0.0 && layer<(numLayers-1))
            {
                // printf( "doing RELU on layer %u\n", layer );
                tempVal = 0.0;
            }
            nnet->temp[i]=tempVal;
        }

        //Output of one layer is the input to the next layer
        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            nnet->inputs[i] = nnet->temp[i];
        }
    }

    //Write the final output value to the allocated spot in memory
    for (i=0; i<outputSize; i++)
    {
        if ( normalizeOutput )
            output[i] = nnet->inputs[i]*nnet->ranges[nnet->inputSize]+nnet->means[nnet->inputSize];
        else
            output[i] = nnet->inputs[i];
    }

    //Return 1 for success
    return 1;
}

//Return the number of inputs to a network
//Inputs: void *network - pointer to a network struct
//Output: int - number of inputs to the network, -1 if the network is NULL
int num_inputs(void *network)
{
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    NNet *nnet = static_cast<NNet*>(network);
    return nnet->inputSize;
}

//Return the number of outputs from a network
//Inputs: void *network - pointer to a network struct
//Output: int - number of outputs from the network, -1 if the network is NULL
int num_outputs(void *network)
{
    if (network ==NULL)
    {
        printf("Data is Null!\n");
        return -1;
    }
    NNet *nnet = static_cast<NNet *>(network);
    return nnet->outputSize;
}