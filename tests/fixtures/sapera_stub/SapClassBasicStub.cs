// Test-only stand-in for Teledyne DALSA DALSA.SaperaLT.SapClassBasic.dll.
//
// It reproduces the public shapes that devices/sapera_api.py calls (overloads, out parameters,
// nested enums, events raised from background threads, IntPtr ReadRect) so that the real
// pythonnet interop can be exercised on a machine without Sapera LT. It is compiled at test time
// with the .NET Framework csc.exe (C# 5 syntax only) and never shipped. Behaviour is scripted by
// StubHardware; it is not a model of real frame-grabber timing.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace DALSA.SaperaLT.SapClassBasic
{
    public static class StubHardware
    {
        public static readonly object Sync = new object();
        public static readonly List<string> Calls = new List<string>();
        public static string[] Servers = new string[] { "System", "Xtium-CL_MX4_1" };
        public static int AcqCount = 1;
        public static int AcqDeviceCount = 1;
        public static bool SignalPresent = true;
        public static bool ScatterGatherSupported = true;
        public static int Width = 64;
        public static int Height = 16;
        public static int PixelDepth = 8;
        public static int ExtraPitch = 0;
        public static int SnapDelayMilliseconds = 20;
        public static bool FailAcquisitionCreate = false;
        public static string[] ReadOnlyFeatures = new string[0];
        public static string[] MissingFeatures = new string[0];
        public static string[] MissingParameters = new string[0];
        // Linea 16K reports AcquisitionLineRate 300..48000 Hz (field report); defaults keep 30 Hz legal.
        public static long LineRateMin = 1;
        public static long LineRateMax = 48000;
        public static SapAcquisition LastAcquisition;
        public static int CallbackThreadId;

        public static void Reset()
        {
            lock (Sync)
            {
                Calls.Clear();
                Servers = new string[] { "System", "Xtium-CL_MX4_1" };
                AcqCount = 1;
                AcqDeviceCount = 1;
                SignalPresent = true;
                ScatterGatherSupported = true;
                Width = 64;
                Height = 16;
                PixelDepth = 8;
                ExtraPitch = 0;
                SnapDelayMilliseconds = 20;
                FailAcquisitionCreate = false;
                ReadOnlyFeatures = new string[0];
                MissingFeatures = new string[0];
                MissingParameters = new string[0];
                LineRateMin = 1;
                LineRateMax = 48000;
                LastAcquisition = null;
                CallbackThreadId = 0;
                SapAcqDevice.Features.Clear();
            }
        }

        public static void Record(string call)
        {
            lock (Sync)
            {
                Calls.Add(call);
            }
        }

        public static string[] TakeCalls()
        {
            lock (Sync)
            {
                var copy = Calls.ToArray();
                Calls.Clear();
                return copy;
            }
        }

        public static void RaiseExternalTrigger()
        {
            var acquisition = LastAcquisition;
            var thread = new Thread(() =>
            {
                CallbackThreadId = Thread.CurrentThread.ManagedThreadId;
                acquisition.RaiseAcqNotify(SapAcquisition.AcqEventType.ExternalTrigger);
            });
            thread.Start();
            thread.Join();
        }

        internal static bool Contains(string[] values, string value)
        {
            return Array.IndexOf(values, value) >= 0;
        }
    }

    public class SapLocation
    {
        public SapLocation(string serverName, int resourceIndex)
        {
            ServerName = serverName;
            ResourceIndex = resourceIndex;
            ServerIndex = Array.IndexOf(StubHardware.Servers, serverName);
        }

        public SapLocation(int serverIndex, int resourceIndex)
        {
            ServerIndex = serverIndex;
            ResourceIndex = resourceIndex;
            ServerName = StubHardware.Servers[serverIndex];
        }

        public string ServerName { get; private set; }
        public int ServerIndex { get; private set; }
        public int ResourceIndex { get; private set; }
    }

    public class SapManager
    {
        public enum ResourceType
        {
            Acq = 0,
            AcqDevice = 1,
            Display = 2,
        }

        public static int GetServerCount()
        {
            return StubHardware.Servers.Length;
        }

        public static string GetServerName(int serverIndex)
        {
            return StubHardware.Servers[serverIndex];
        }

        public static int GetResourceCount(string serverName, ResourceType resourceType)
        {
            if (serverName == "System")
            {
                return 0;
            }

            return resourceType == ResourceType.Acq ? StubHardware.AcqCount
                : resourceType == ResourceType.AcqDevice ? StubHardware.AcqDeviceCount : 0;
        }

        public static int GetResourceCount(int serverIndex, ResourceType resourceType)
        {
            return GetResourceCount(StubHardware.Servers[serverIndex], resourceType);
        }

        public static string GetResourceName(string serverName, ResourceType resourceType, int resourceIndex)
        {
            return (resourceType == ResourceType.Acq ? "CameraLink Mono #" : "Camera #") + (resourceIndex + 1);
        }

        public static bool IsResourceAvailable(string serverName, ResourceType resourceType, int resourceIndex)
        {
            return true;
        }
    }

    public class SapFeature : IDisposable
    {
        public enum AccessMode
        {
            Undefined = 0,
            ReadOnly = 1,
            WriteOnly = 2,
            ReadWrite = 3,
        }

        public SapFeature(SapLocation location)
        {
            DataAccessMode = AccessMode.Undefined;
        }

        public bool Initialized { get; private set; }
        public AccessMode DataAccessMode { get; internal set; }
        internal string Name;

        // Both widths exist so the binding has to pick the Int64 overload explicitly.
        public bool GetValueMin(out int value)
        {
            value = (int)StubHardware.LineRateMin;
            return Name == "AcquisitionLineRate";
        }

        public bool GetValueMin(out long value)
        {
            value = StubHardware.LineRateMin;
            return Name == "AcquisitionLineRate";
        }

        public bool GetValueMax(out int value)
        {
            value = (int)StubHardware.LineRateMax;
            return Name == "AcquisitionLineRate";
        }

        public bool GetValueMax(out long value)
        {
            value = StubHardware.LineRateMax;
            return Name == "AcquisitionLineRate";
        }

        public bool Create()
        {
            Initialized = true;
            return true;
        }

        public bool Destroy()
        {
            Initialized = false;
            return true;
        }

        public void Dispose()
        {
            Initialized = false;
        }
    }

    public class SapAcqDevice : IDisposable
    {
        internal static readonly Dictionary<string, string> Features = new Dictionary<string, string>();

        public SapAcqDevice(SapLocation location)
        {
            Location = location;
        }

        public SapLocation Location { get; private set; }
        public bool Initialized { get; private set; }

        public bool Create()
        {
            StubHardware.Record("AcqDevice.Create " + Location.ServerName + "#" + Location.ResourceIndex);
            Initialized = true;
            return true;
        }

        public bool Destroy()
        {
            StubHardware.Record("AcqDevice.Destroy");
            Initialized = false;
            return true;
        }

        public void Dispose()
        {
            StubHardware.Record("AcqDevice.Dispose");
            Initialized = false;
        }

        public bool IsFeatureAvailable(string featureName)
        {
            return !StubHardware.Contains(StubHardware.MissingFeatures, featureName);
        }

        public bool GetFeatureInfo(string featureName, SapFeature feature)
        {
            if (!IsFeatureAvailable(featureName))
            {
                return false;
            }

            feature.Name = featureName;
            feature.DataAccessMode = StubHardware.Contains(StubHardware.ReadOnlyFeatures, featureName)
                ? SapFeature.AccessMode.ReadOnly
                : SapFeature.AccessMode.ReadWrite;
            return true;
        }

        private bool Store(string featureName, string value, string kind)
        {
            StubHardware.Record("SetFeatureValue(" + kind + ") " + featureName + "=" + value);
            if (!IsFeatureAvailable(featureName) || StubHardware.Contains(StubHardware.ReadOnlyFeatures, featureName))
            {
                return false;
            }

            lock (StubHardware.Sync)
            {
                Features[featureName] = value;
            }

            return true;
        }

        public bool SetFeatureValue(string featureName, string value) { return Store(featureName, value, "String"); }
        public bool SetFeatureValue(string featureName, long value) { return Store(featureName, value.ToString(), "Int64"); }
        public bool SetFeatureValue(string featureName, int value) { return Store(featureName, value.ToString(), "Int32"); }
        public bool SetFeatureValue(string featureName, double value) { return Store(featureName, value.ToString(System.Globalization.CultureInfo.InvariantCulture), "Double"); }
        public bool SetFeatureValue(string featureName, bool value) { return Store(featureName, value.ToString(), "Boolean"); }

        public bool GetFeatureValue(string featureName, out string value)
        {
            lock (StubHardware.Sync)
            {
                return Features.TryGetValue(featureName, out value);
            }
        }

        public bool GetFeatureValue(string featureName, out long value)
        {
            string text;
            value = 0;
            return GetFeatureValue(featureName, out text) && long.TryParse(text, out value);
        }

        public bool GetFeatureValue(string featureName, out int value)
        {
            string text;
            value = 0;
            return GetFeatureValue(featureName, out text) && int.TryParse(text, out value);
        }

        public bool GetFeatureValue(string featureName, out double value)
        {
            string text;
            value = 0;
            return GetFeatureValue(featureName, out text) && double.TryParse(text, out value);
        }

        public bool UpdateFeaturesToDevice()
        {
            StubHardware.Record("UpdateFeaturesToDevice");
            return true;
        }
    }

    public class SapAcqNotifyEventArgs : EventArgs
    {
        public SapAcqNotifyEventArgs(SapAcquisition.AcqEventType eventType) { EventType = eventType; }
        public SapAcquisition.AcqEventType EventType { get; private set; }
    }

    public class SapSignalNotifyEventArgs : EventArgs
    {
        public SapSignalNotifyEventArgs(SapAcquisition.AcqSignalStatus status) { SignalStatus = status; }
        public SapAcquisition.AcqSignalStatus SignalStatus { get; private set; }
    }

    public class SapXferNotifyEventArgs : EventArgs
    {
        public SapXferNotifyEventArgs(bool trash) { Trash = trash; }
        public bool Trash { get; private set; }
    }

    public delegate void SapAcqNotifyHandler(object sender, SapAcqNotifyEventArgs argsNotify);
    public delegate void SapSignalNotifyHandler(object sender, SapSignalNotifyEventArgs argsSignal);
    public delegate void SapXferNotifyHandler(object sender, SapXferNotifyEventArgs argsNotify);

    public class SapAcqCamIoControl
    {
        public int Value { get; set; }
    }

    // Real Sapera declares buffer sources as SapXferNode; SapAcquisition derives from it, which is why
    // `new SapBufferWithTrash(2, acquisition, ...)` compiles in the reference app (field report 050503).
    public abstract class SapXferNode
    {
    }

    public class SapAcquisition : SapXferNode, IDisposable
    {
        public enum Prm
        {
            CROP_HEIGHT,
            LINE_INTEGRATE_ENABLE,
            LINE_INTEGRATE_METHOD,
            LINE_INTEGRATE_DURATION,
            LINE_INTEGRATE_PULSE0_POLARITY,
            LINE_INTEGRATE_PULSE1_POLARITY,
            LINE_TRIGGER_METHOD,
            LINE_TRIGGER_ENABLE,
            INT_LINE_TRIGGER_ENABLE,
            INT_LINE_TRIGGER_FREQ,
            INT_LINE_TRIGGER_FREQ_MIN,
            INT_LINE_TRIGGER_FREQ_MAX,
            CAM_LINE_TRIGGER_FREQ_MIN,
            CAM_LINE_TRIGGER_FREQ_MAX,
            EXT_LINE_TRIGGER_ENABLE,
            EXT_FRAME_TRIGGER_ENABLE,
            INT_FRAME_TRIGGER_ENABLE,
            SHAFT_ENCODER_ENABLE,
            CAM_TRIGGER_ENABLE,
            EXT_TRIGGER_ENABLE,
        }

        public enum Val
        {
            LINE_INTEGRATE_METHOD_3 = 4,
            ACTIVE_HIGH = 8,
            ACTIVE_LOW = 16,
            SIGNAL_NAME_PULSE1 = 0x20000,
        }

        public enum Cap
        {
            LINE_TRIGGER_METHOD,
        }

        [Flags]
        public enum AcqEventType
        {
            None = 0,
            ExternalTrigger = 1,
            ExternalTrigger2 = 2,
            ExternalTriggerIgnored = 4,
            ExternalTriggerTooSlow = 8,
            ExtLineTriggerTooSlow = 16,
            LineTriggerTooFast = 32,
        }

        public enum AcqSignalStatus
        {
            None = 0,
            HSyncPresent = 1,
            PixelClkPresent = 2,
        }

        private readonly Dictionary<Prm, int> _parameters = new Dictionary<Prm, int>();
        private SapAcqCamIoControl[] _camIo = new SapAcqCamIoControl[] { new SapAcqCamIoControl(), new SapAcqCamIoControl() };

        public SapAcquisition(SapLocation location, string configFile)
        {
            Location = location;
            ConfigFile = configFile;
            StubHardware.LastAcquisition = this;
            _parameters[Prm.CROP_HEIGHT] = StubHardware.Height;
            _parameters[Prm.INT_LINE_TRIGGER_FREQ_MIN] = 100;
            _parameters[Prm.INT_LINE_TRIGGER_FREQ_MAX] = 50000;
        }

        public SapLocation Location { get; private set; }
        public string ConfigFile { get; private set; }
        public bool Initialized { get; private set; }
        public event SapAcqNotifyHandler AcqNotify;
        public event SapSignalNotifyHandler SignalNotify;
        public object AcqNotifyContext { get; set; }
        public object SignalNotifyContext { get; set; }
        public AcqEventType EventType { get; set; }
        public bool SignalNotifyEnable { get; set; }

        public AcqSignalStatus SignalStatus
        {
            get { return StubHardware.SignalPresent ? AcqSignalStatus.HSyncPresent : AcqSignalStatus.None; }
        }

        public SapAcqCamIoControl[] CamIoControl
        {
            get
            {
                var copy = new SapAcqCamIoControl[_camIo.Length];
                for (var i = 0; i < copy.Length; i++)
                {
                    copy[i] = new SapAcqCamIoControl();
                    copy[i].Value = _camIo[i].Value;
                }

                return copy;
            }
            set
            {
                StubHardware.Record("CamIoControl[0]=" + value[0].Value);
                _camIo = value;
            }
        }

        public bool Create()
        {
            StubHardware.Record("Acquisition.Create " + ConfigFile);
            Initialized = !StubHardware.FailAcquisitionCreate;
            return Initialized;
        }

        public bool Destroy()
        {
            StubHardware.Record("Acquisition.Destroy");
            Initialized = false;
            return true;
        }

        public void Dispose()
        {
            StubHardware.Record("Acquisition.Dispose");
            Initialized = false;
        }

        public bool IsParameterAvailable(Prm parameter)
        {
            return !StubHardware.Contains(StubHardware.MissingParameters, parameter.ToString());
        }

        public bool GetParameter(Prm parameter, out int value)
        {
            value = 0;
            if (!IsParameterAvailable(parameter))
            {
                return false;
            }

            if (!_parameters.TryGetValue(parameter, out value))
            {
                value = 0;
            }

            return true;
        }

        // Extra overloads so the interop must select (Prm, out Int32) explicitly.
        public bool GetParameter(Prm parameter, out IntPtr value)
        {
            value = IntPtr.Zero;
            return false;
        }

        public bool GetParameter(Prm parameter, out double value)
        {
            value = 0;
            return false;
        }

        public bool SetParameter(Prm parameter, int value, bool updateNow)
        {
            StubHardware.Record("SetParameter " + parameter + "=" + value);
            if (!IsParameterAvailable(parameter))
            {
                return false;
            }

            _parameters[parameter] = value;
            return true;
        }

        public bool SetParameter(Prm parameter, Val value, bool updateNow)
        {
            StubHardware.Record("SetParameter " + parameter + "=" + value);
            if (!IsParameterAvailable(parameter))
            {
                return false;
            }

            _parameters[parameter] = (int)value;
            return true;
        }

        public bool SetParameter(Prm parameter, IntPtr value, bool updateNow)
        {
            StubHardware.Record("SetParameter(IntPtr) " + parameter);
            return false;
        }

        public bool GetCapability(Cap capability, out int value)
        {
            value = 0x6;
            return true;
        }

        internal void RaiseAcqNotify(AcqEventType eventType)
        {
            var handler = AcqNotify;
            if (handler != null)
            {
                handler(this, new SapAcqNotifyEventArgs(eventType));
            }
        }

        internal void RaiseSignalNotify(AcqSignalStatus status)
        {
            var handler = SignalNotify;
            if (handler != null)
            {
                handler(this, new SapSignalNotifyEventArgs(status));
            }
        }
    }

    public class SapBuffer : IDisposable
    {
        public enum MemoryType
        {
            Default = 0,
            ScatterGather = 1,
            ScatterGatherPhysical = 2,
        }

        public enum Prm
        {
            PIXEL_DEPTH,
            PITCH,
            FORMAT,
        }

        protected SapBuffer() { }

        public bool Initialized { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        internal int FrameIndex;

        public static bool IsBufferTypeSupported(SapLocation location, MemoryType memoryType)
        {
            return memoryType != MemoryType.ScatterGather || StubHardware.ScatterGatherSupported;
        }

        public bool Create()
        {
            StubHardware.Record("Buffer.Create");
            Width = StubHardware.Width;
            Height = StubHardware.Height;
            Initialized = true;
            return true;
        }

        public bool Clear()
        {
            return true;
        }

        public bool Destroy()
        {
            StubHardware.Record("Buffer.Destroy");
            Initialized = false;
            return true;
        }

        public void Dispose()
        {
            StubHardware.Record("Buffer.Dispose");
            Initialized = false;
        }

        public bool GetParameter(Prm parameter, out int value)
        {
            value = parameter == Prm.PIXEL_DEPTH ? StubHardware.PixelDepth
                : parameter == Prm.PITCH ? StubHardware.Width + StubHardware.ExtraPitch : 0;
            return true;
        }

        public bool GetParameter(Prm parameter, out IntPtr value)
        {
            value = IntPtr.Zero;
            return false;
        }

        // Writes rows at the buffer pitch, as xx_ccd CameraService.TryCreatePreviewBitmap expects.
        public bool ReadRect(int x, int y, int width, int height, IntPtr destination)
        {
            var pitch = StubHardware.Width + StubHardware.ExtraPitch;
            var row = new byte[pitch];
            for (var r = 0; r < height; r++)
            {
                for (var c = 0; c < pitch; c++)
                {
                    row[c] = c < width ? (byte)((r * 7 + c + FrameIndex) & 0xFF) : (byte)0xEE;
                }

                Marshal.Copy(row, 0, destination + (r * pitch), pitch);
            }

            return true;
        }
    }

    public class SapBufferWithTrash : SapBuffer
    {
        public SapBufferWithTrash(int count, SapXferNode srcNode, MemoryType memoryType)
        {
            StubHardware.Record("SapBufferWithTrash(" + count + "," + memoryType + ")");
        }
    }

    public class SapXferPair
    {
        [Flags]
        public enum XferEventType
        {
            None = 0,
            StartOfFrame = 1,
            EndOfFrame = 2,
        }

        public XferEventType EventType { get; set; }
    }

    public class SapAcqToBuf : IDisposable
    {
        private readonly SapAcquisition _acquisition;
        private readonly SapBuffer _buffer;
        private volatile bool _grabbing;
        private Thread _worker;

        public SapAcqToBuf(SapAcquisition acquisition, SapBuffer buffer)
        {
            _acquisition = acquisition;
            _buffer = buffer;
            Pairs = new SapXferPair[] { new SapXferPair() };
        }

        public SapXferPair[] Pairs { get; private set; }
        public event SapXferNotifyHandler XferNotify;
        public object XferNotifyContext { get; set; }
        public bool Initialized { get; private set; }

        public bool Create()
        {
            StubHardware.Record("Transfer.Create EventType=" + Pairs[0].EventType);
            Initialized = true;
            return true;
        }

        public bool Destroy()
        {
            StubHardware.Record("Transfer.Destroy");
            Initialized = false;
            return true;
        }

        public void Dispose()
        {
            StubHardware.Record("Transfer.Dispose");
            Initialized = false;
        }

        public bool Snap()
        {
            StubHardware.Record("Snap");
            _worker = new Thread(() =>
            {
                Thread.Sleep(StubHardware.SnapDelayMilliseconds);
                DeliverFrame();
            });
            _worker.Start();
            return true;
        }

        public bool Grab()
        {
            StubHardware.Record("Grab");
            _grabbing = true;
            _worker = new Thread(() =>
            {
                while (_grabbing)
                {
                    Thread.Sleep(StubHardware.SnapDelayMilliseconds);
                    if (_grabbing)
                    {
                        DeliverFrame();
                    }
                }
            });
            _worker.Start();
            return true;
        }

        public bool Freeze()
        {
            StubHardware.Record("Freeze");
            _grabbing = false;
            return true;
        }

        public bool Wait(int timeout)
        {
            var worker = _worker;
            return worker == null || worker.Join(timeout);
        }

        private void DeliverFrame()
        {
            StubHardware.CallbackThreadId = Thread.CurrentThread.ManagedThreadId;
            _buffer.FrameIndex++;
            var handler = XferNotify;
            if (handler != null)
            {
                handler(this, new SapXferNotifyEventArgs(false));
            }
        }
    }
}
