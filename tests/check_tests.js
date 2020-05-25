const fs = require('fs');
const xml2js = require('xml2js');

let TESTS_DIR = 'features';

const testsDirOption = process.argv.indexOf('--tests-dir');

if (testsDirOption !== -1 && process.argv[testsDirOption + 1]) {
    TESTS_DIR = process.argv[testsDirOption + 1];
}

let allTests = {};

function readTestResults(testDir, callback) {
    let fileCount = 0;
    fs.readdir(testDir, function (err, testFolders) {
        if (err) console.log(err.message);
        else {
            testFolders.forEach(function (folderName) {
                if (fs.lstatSync(testDir + '/' + folderName).isDirectory()) {
                    const testPath = testDir + '/' + folderName;
                    if (fs.existsSync(testPath + '/test_results.xml')) {
                        fs.readFile(testPath + '/test_results.xml', function (err, data) {
                            xml2js.parseString(data, function (err, xmlData) {
                                if (err) console.log(err);
                                allTests[testPath] = parseInt(xmlData['testsuites']['$']['failures']);
                                if (Object.keys(allTests).length === (testFolders.length - fileCount)) {
                                    callback(allTests);
                                }
                            });
                        });
                    } else {
                        allTests[testPath] = 'NONE';
                        if (Object.keys(allTests).length === (testFolders.length - fileCount)) {
                            callback(allTests);
                        }
                    }
                } else {
                    fileCount++;
                }
            });
        }
    });
}

readTestResults(TESTS_DIR, function (allTests) {
    let total = Object.keys(allTests).length;
    let passed = Object.keys(allTests).filter(testName => allTests[testName] === 0);
    let failed = Object.keys(allTests).filter(testName => allTests[testName] === 1);
    let noResult = Object.keys(allTests).filter(testName => allTests[testName] === 'NONE');

    console.log('//--------------------------');
    console.log('// PASSED');
    console.log('//--------------------------');
    passed.forEach(test => console.log('PASSED: ' + test.replace('features//', '')));
    console.log('//--------------------------');
    console.log('// FAILED');
    console.log('//--------------------------');
    failed.forEach(test => console.log('FAILED: ' + test.replace('features//', '')));
    console.log('//--------------------------');
    console.log('// NO RESULT (Compiler Errors)');
    console.log('//--------------------------');
    noResult.forEach(test => console.log('NO RESULT: ' + test.replace('features//', '')));


    console.log('\n\n');
    console.log('//--------------------------');
    console.log('// SUMMARY');
    console.log('//--------------------------');


    console.log('TOTAL: \t\t' + total + '\t100%');
    console.log('PASSED: \t' + passed.length + '\t' + parseInt(passed.length / total * 100) + '%');
    console.log('FAILED: \t' + failed.length + '\t' + parseInt(failed.length / total * 100) + '%');
    console.log('NO RESULT: \t' + noResult.length + '\t' + parseInt(noResult.length / total * 100) + '%');
});